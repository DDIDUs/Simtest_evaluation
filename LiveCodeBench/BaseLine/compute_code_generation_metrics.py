import os
import sys

sys.set_int_max_str_digits(50000)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
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
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    # 입력 케이스 수에 비례한 글로벌 타임아웃
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

    # 최소 1개 result/metadata 존재
    return result[0], metadata_list[0]


def evaluate_generations_by_problem(args):
    """
    문제 하나(qid 하나)에 대해 여러 generation을 평가.
    반환값:
      - res: 각 generation에 대한 테스트 결과 리스트 (list[list[bool/int]])
      - metadata: 각 generation에 대한 메타데이터 리스트 (list[dict])
    """
    problem_generations: list[str] = args[0]
    sample = args[1]
    debug: bool = args[2]
    timeout: int = args[3]

    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res, curr_metadata = check_correctness(
                sample, o, timeout=timeout, debug=debug
            )
            if debug:
                print(f"\nSuccessful compilation of task {o_idx}!")
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    print(f"Results were not True for all test cases {curr_res=}\n")
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            curr_metadata = {
                "error": repr(e),
                "error_code": -5,
                "error_message": "TestRunnerError",
            }
        finally:
            assert isinstance(curr_res, list), curr_res
            assert isinstance(curr_metadata, dict), curr_metadata
            res.append(curr_res)
            metadata.append(curr_metadata)

    if debug:
        for i, r in enumerate(problem_generations):
            print("Sample\n")
            print(r)
            print("\n")
            print("Result\n")
            print(res[i])
            print("*" * 30 + "\n\n")

    return res, metadata


def evaluate_generations(
    samples_list: list,
    generations_by_qid: dict,   # 변경: list[list[str]] → dict[qid -> list[str]]
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
):

    # qid -> sample 매핑
    samples_by_qid = {}
    for sample in samples_list:
        qid = sample.get("question_id")
        if qid is None:
            # 필요하면 여기서 에러로 바꿔도 됨
            in_outs = json.loads(sample["input_output"])
            qid = in_outs.get("question_id")
        if qid is None:
            raise ValueError("sample에 question_id가 없습니다.")
        samples_by_qid[str(qid)] = sample

    # 공통 qid 집합 (테스트와 코드가 모두 있는 qid만 평가)
    common_qids = sorted(
        set(samples_by_qid.keys()) & set(str(k) for k in generations_by_qid.keys())
    )
    if not common_qids:
        raise ValueError("samples와 generations 사이에 공통 question_id가 없습니다.")

    # 프로세스로 넘길 입력 구성: (generations, sample, debug, timeout), qid
    inputs = []
    for qid in common_qids:
        gens = generations_by_qid[str(qid)]
        sample = samples_by_qid[str(qid)]
        inputs.append(((gens, sample, debug, timeout), qid))

    with tqdm(total=len(inputs)) as pbar:
        with ProcessPoolExecutor(
            max_workers=1 if debug else num_process_evaluate
        ) as executor:
            futures = {
                executor.submit(evaluate_generations_by_problem, arg): qid
                for arg, qid in inputs
            }

            results = {}
            metadata = {}
            for future in as_completed(futures):
                qid = futures[future]  # 여기서부터 key는 순수 qid
                res, meta = future.result()
                print(res)
                results[str(qid)] = res
                metadata[str(qid)] = meta
                pbar.update(1)

    assert len(results) == len(
        inputs
    ), f"results = {len(results)} inputs = {len(inputs)} {results=}"

    return results, metadata
