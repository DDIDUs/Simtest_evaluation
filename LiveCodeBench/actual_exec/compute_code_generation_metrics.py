# compute_code_generation_metrics.py

import os
import sys

sys.set_int_max_str_digits(50000)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from testing_util import run_test


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(sample, generation, timeout, debug=True):
    """
    run_test는 sample["input_output"]에 포함된 테스트들을 수행한다.
    이번 스펙에서는 sample을 "테스트 1개짜리"로 만들어 이 함수를 반복 호출한다.
    """
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()

    in_outs = json.loads(sample["input_output"])
    num_tests = len(in_outs["inputs"])
    p.join(timeout=(timeout + 1) * num_tests + 5)

    if p.is_alive():
        p.kill()

    if not result:
        # 글로벌 타임아웃 시, 테스트 개수만큼 -1 생성
        result = [[-1 for _ in range(num_tests)]]
        metadata_list.append(
            {
                "error": "GlobalTimeout",
                "error_code": -6,
                "error_message": "Global timeout exceeded in check_correctness",
            }
        )
        if debug:
            print("global timeout")

    return result[0], metadata_list[0]


def _normalize_bool(x):
    if isinstance(x, np.ndarray):
        x = x.item(0)
    if isinstance(x, np.bool_):
        x = bool(x)
    return x


def _build_single_test_sample(base_sample, one_input, one_output, fn_name):
    return {
        "input_output": json.dumps(
            {
                "inputs": [one_input],
                "outputs": [one_output],
                "fn_name": fn_name,
            }
        ),
        "question_id": base_sample.get("question_id"),
    }


def _select_unique_extreme(test_ids, pass_count_by_test, mode="max"):
    """
    easy/hard 규칙:
      - 유일 max/min인 테스트만 선정
      - 동률이면 빈 리스트 + None
    """
    if not test_ids:
        return [], None

    vals = [pass_count_by_test[tid] for tid in test_ids]
    extreme = max(vals) if mode == "max" else min(vals)
    cands = [tid for tid in test_ids if pass_count_by_test[tid] == extreme]

    if len(cands) == 1:
        return cands, extreme
    return [], None


def _select_medium_one(
    test_ids,
    pass_count_by_test,
    easy_ids, easy_pc,
    hard_ids, hard_pc,
    qid_for_seed: str,
):
    """
    medium 규칙 (요구 반영):
      1) hard < pass_count < easy 범위 중 1개를 랜덤 선택 (easy/hard 제외)
      2) 없으면 (easy/hard 제외한) 나머지 중 easy_pass_count에 가장 가까운 쪽(차이 최소)에서 랜덤 1개
      3) 후보 없으면 [] / None

    랜덤은 qid 기반으로 고정(seed)하여 재현 가능하게 함.
    """
    rng = random.Random(qid_for_seed)

    excluded = set(easy_ids) | set(hard_ids)
    remaining = [tid for tid in test_ids if tid not in excluded]
    if not remaining:
        return [], None

    # easy/hard가 유일 선정 실패(None)인 경우에도 medium을 고르기 위해 기준값 보정
    if easy_pc is None:
        easy_pc = max(pass_count_by_test[tid] for tid in test_ids)
    if hard_pc is None:
        hard_pc = min(pass_count_by_test[tid] for tid in test_ids)

    low, high = sorted([hard_pc, easy_pc])

    # 1) strict between 후보
    between = [tid for tid in remaining if low < pass_count_by_test[tid] < high]
    if between:
        chosen = rng.choice(between)
        return [chosen], pass_count_by_test[chosen]

    # 2) 없으면 easy에 가장 가까운 쪽 (abs diff 최소)에서 랜덤
    diffs = {tid: abs(pass_count_by_test[tid] - easy_pc) for tid in remaining}
    min_diff = min(diffs.values())
    closest = [tid for tid in remaining if diffs[tid] == min_diff]

    chosen = rng.choice(closest)
    return [chosen], pass_count_by_test[chosen]


def evaluate_generations_by_problem(args):
    """
    문제(row) 단위 처리:
      - 한 문제에 포함된 모든 코드에 대해:
        - 모든 테스트 케이스를 1개씩 개별 실행
        - eval_result: list[list[bool]] 로만 결과 저장
      - difficulty는 eval_result를 종합해 계산
        - easy/hard: 유일 max/min만 선택 (동률이면 비움)
        - medium: 중간 범위 랜덤 1개, 없으면 easy에 가까운 쪽 랜덤 1개
      - difficulty에는 테스트 "id"가 아니라 테스트 케이스 객체를 저장:
        {"id": "...", "input": ..., "output": ...}
      - pass_count_by_test는 출력에 저장하지 않음(내부 계산용만 사용)
    """
    problem_generations: list[str] = args[0]
    sample = args[1]
    debug: bool = args[2]
    timeout: int = args[3]

    in_outs = json.loads(sample["input_output"])
    inputs = in_outs["inputs"]
    outputs = in_outs["outputs"]
    fn_name = in_outs.get("fn_name", None)

    num_tests = len(inputs)
    test_ids = [str(i) for i in range(num_tests)]

    # 테스트 케이스 "객체" 저장 (difficulty에 이것을 넣음)
    test_cases = [
        {"id": str(i), "input": inputs[i], "output": outputs[i]}
        for i in range(num_tests)
    ]
    case_by_id = {tc["id"]: tc for tc in test_cases}

    # eval_result: 코드별 [테스트별 bool...]
    eval_result: list[list[bool]] = []

    for code_idx, code in enumerate(problem_generations):
        per_code_result: list[bool] = []

        for t_idx, (one_in, one_out) in enumerate(zip(inputs, outputs)):
            one_sample = _build_single_test_sample(sample, one_in, one_out, fn_name)

            passed = False
            try:
                res, _meta = check_correctness(one_sample, code, timeout=timeout, debug=debug)
                # one_sample은 테스트 1개이므로 res[0]만 확인
                if isinstance(res, list) and len(res) >= 1:
                    r0 = _normalize_bool(res[0])
                    passed = (r0 is True)
            except Exception as e:
                if debug:
                    print(f"[qid={sample.get('question_id')}] code#{code_idx} test#{t_idx} exception: {repr(e)}")
                passed = False

            per_code_result.append(passed)

        eval_result.append(per_code_result)

    # pass_count 계산 (출력 저장 금지)
    pass_count_by_test = {
        tid: sum(1 for c in eval_result if c[int(tid)] is True)
        for tid in test_ids
    }

    # easy/hard: 유일 max/min만
    easy_ids, easy_pc = _select_unique_extreme(test_ids, pass_count_by_test, mode="max")
    hard_ids, hard_pc = _select_unique_extreme(test_ids, pass_count_by_test, mode="min")

    # medium: 중간 후보 랜덤 1개, 없으면 easy 가까운 쪽 랜덤 1개
    qid_seed = str(sample.get("question_id", ""))
    medium_ids, medium_pc = _select_medium_one(
        test_ids=test_ids,
        pass_count_by_test=pass_count_by_test,
        easy_ids=easy_ids,
        easy_pc=easy_pc,
        hard_ids=hard_ids,
        hard_pc=hard_pc,
        qid_for_seed=qid_seed,
    )

    difficulty = {
        "easy_tests": [case_by_id[tid] for tid in easy_ids],
        "medium_tests": [case_by_id[tid] for tid in medium_ids],
        "hard_tests": [case_by_id[tid] for tid in hard_ids],
        "easy_pass_count": easy_pc,
        "medium_pass_count": medium_pc,
        "hard_pass_count": hard_pc,
    }

    return eval_result, difficulty


def evaluate_generations(
    samples_list: list,
    generations_by_qid: dict,  # dict[qid -> list[str]]
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
    show_progress: bool = True,
):
    """
    샘플 1개(qid 1개) 완료될 때마다 (qid, eval_result, difficulty)를 yield.
    상위 레벨에서 이걸 받아서 즉시 파일에 저장하면 됨.
    """
    # qid -> sample 매핑
    samples_by_qid = {}
    for sample in samples_list:
        qid = sample.get("question_id")
        if qid is None:
            in_outs = json.loads(sample["input_output"])
            qid = in_outs.get("question_id")
        if qid is None:
            raise ValueError("sample에 question_id가 없습니다.")
        samples_by_qid[str(qid)] = sample

    common_qids = sorted(set(samples_by_qid.keys()) & set(str(k) for k in generations_by_qid.keys()))
    if not common_qids:
        raise ValueError("samples와 generations 사이에 공통 question_id가 없습니다.")

    jobs = []
    for qid in common_qids:
        gens = generations_by_qid[str(qid)]
        sample = samples_by_qid[str(qid)]
        jobs.append(((gens, sample, debug, timeout), qid))

    pbar = tqdm(total=len(jobs)) if show_progress else None

    with ProcessPoolExecutor(max_workers=1 if debug else num_process_evaluate) as executor:
        futures = {
            executor.submit(evaluate_generations_by_problem, arg): qid
            for arg, qid in jobs
        }

        for future in as_completed(futures):
            qid = futures[future]
            eval_result, difficulty = future.result()

            if pbar is not None:
                pbar.update(1)

            yield str(qid), eval_result, difficulty

    if pbar is not None:
        pbar.close()
