# borrowed and extended from
# https://github.com/Naman-ntc/codescratch/blob/main/evaluation/bigcode-evaluation-harness/lm_eval/tasks/custom_metrics/apps_custom_metrics/utils.py

import os
import sys

sys.set_int_max_str_digits(50000)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import multiprocessing
from collections import defaultdict
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
    sample["input_output"] 는 다음 형태의 JSON string 이라고 가정:
      {
        "inputs": [...],
        "outputs": [...],
        "fn_name": "..."
      }
    """

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
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
):
    """
    samples_list 의 각 sample 과 generations_list 의 각 코드에 대해
    테스트를 수행하고, 각 generation 에 대해 모든 테스트 결과를 수집한다.

    반환 형식:
      results = {
        "<index>_<fn_name>": [[...], [...], ...],   # 문제별 여러 generation 결과
        ...
      }
      metadata = {
        "<index>_<fn_name>": [ {...}, {...}, ... ],
        ...
      }
    """

    assert len(samples_list) == len(
        generations_list
    ), f"samples_list and generations_list length mismatch: {len(samples_list)} vs {len(generations_list)}"

    # index -> "{index}_{fn_name}" 매핑 미리 계산
    index_to_key = {}
    for idx, sample in enumerate(samples_list):
        in_outs = json.loads(sample["input_output"])
        fn_name = in_outs.get("fn_name")

        # fn_name 없으면 question_id, 그것도 없으면 idx 사용
        if fn_name is None:
            fn_name = sample.get("question_id", idx)

        index_to_key[idx] = f"{idx}_{fn_name}"

    # 프로세스로 넘길 입력 구성 (각 index를 같이 들고 감)
    inputs = [
        [(generations_list[index], samples_list[index], debug, timeout), index]
        for index in range(len(generations_list))
    ]

    with tqdm(total=len(inputs)) as pbar:
        with ProcessPoolExecutor(
            max_workers=1 if debug else num_process_evaluate
        ) as executor:
            futures = {
                executor.submit(evaluate_generations_by_problem, arg): index
                for arg, index in inputs
            }

            results = {}
            metadata = {}
            for future in as_completed(futures):
                index = futures[future]
                key = index_to_key[index]

                res, meta = future.result()
                results[key] = res
                metadata[key] = meta

                pbar.update(1)

    assert len(results) == len(
        inputs
    ), f"results = {len(results)} inputs = {len(inputs)} {results=}"

    return results, metadata
