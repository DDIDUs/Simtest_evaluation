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
    Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`.

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
        # 모든 테스트가 실행되지 못한 (글로벌 타임아웃) 경우에도
        # 테스트 개수만큼 -1 결과를 생성하고 메타데이터를 남긴다.
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

    # 여기까지 오면 최소 1개 result/metadata 가 있음
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
    """

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
                results[index], metadata[index] = future.result()
                pbar.update(1)

    assert len(results) == len(
        inputs
    ), f"results = {len(results)} inputs = {len(inputs)} {results=}"

    return results, metadata


def codegen_metrics(
    samples_list,
    generations_list,
    output_path: str = "results.json",
    num_process_evaluate=16,
    timeout=6,
    debug=False,
):
    """
    samples_list, generations_list 에 대해 테스트를 수행하고
    각 코드 샘플의 모든 테스트 수행 결과를 JSON 파일로 저장한다.

    문제 번호(question_id)는 각 sample 내부에 다음과 같이 들어 있다고 가정:
      eval_sample = {
        "input_output": json.dumps({
            "inputs": inputs,
            "outputs": outputs,
            "fn_name": fn_name,
        }),
        "question_id": row.get("question_id", None),
      }

    저장 형식:
    {
      "results": {
        "<question_id>": [[...], [...], ...],
        ...
      },
      "metadata": {
        "<question_id>": [ {...}, {...}, ... ],
        ...
      }
    }

    question_id 가 없는 경우에는 samples_list 의 인덱스를 fallback key 로 사용.
    """

    samples_linear = []
    generations_linear = []
    remap_question_id = []  # linear index -> question_id
    results = defaultdict(list)
    metadatas = defaultdict(list)

    # (sample, 그 sample 의 여러 generation 리스트)를 선형으로 펼치기
    for idx, (sample, generation_list) in enumerate(
        zip(samples_list, generations_list)
    ):
        assert isinstance(generation_list, list), generations_list[0]

        # 문제 번호는 sample["question_id"] 에 들어 있음
        qid = sample.get("question_id", idx)

        for generation in generation_list:
            assert isinstance(generation, str), generations_list[0]
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_question_id.append(qid)

    print(f"Evaluating {len(samples_linear)} generations...")

    # 각 (sample, 단일 generation)에 대해 테스트 실행
    results_linear, metadatas_linear = evaluate_generations(
        samples_linear,
        generations_linear,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )

    # linear index -> question_id 로 재매핑
    for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        qid = remap_question_id[idx]
        # sub_results: [[테스트 결과들]] 형태이므로 [0]만 취함
        results[qid].append(sub_results[0])

    for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
        qid = remap_question_id[idx]
        # sub_metadatas: [metadata dict] 형태이므로 [0]만 취함
        metadatas[qid].append(sub_metadatas[0])

    # JSON 직렬화를 위해 key를 문자열로 변환
    results_json = {str(k): v for k, v in results.items()}
    metadatas_json = {str(k): v for k, v in metadatas.items()}

    # JSON 파일로 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results_json,
                "metadata": metadatas_json,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 필요하다면 그대로 반환도 해 둠
    return results, metadatas


if __name__ == "__main__":
    print(
        check_correctness(
            {
                "input_output": json.dumps(
                    {
                        "inputs": [")))))"],
                        "outputs": ["0"],
                        "fn_name": "solution"
                    },
                ),
                "question_id": "test",
            },
            "\nMOD = 998244353\n\nS = input().strip()\nn = len(S)\n\nif n % 2 != 0:\n    print(0)\n    exit()\n\n# Initialize DP table\ndp = [[0] * (n + 2) for _ in range(n + 1)]\ndp[0][0] = 1\n\nfor i in range(1, n + 1):\n    c = S[i-1]\n    for b in range(n + 1):\n        if dp[i-1][b] == 0:\n            continue\n        if c == '(':\n            new_b = b + 1\n            if new_b <= n:\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n        elif c == ')':\n            if b > 0:\n                new_b = b - 1\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n        else:  # '?'\n            # Replace with '('\n            new_b = b + 1\n            if new_b <= n:\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n            # Replace with ')'\n            if b > 0:\n                new_b = b - 1\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n\nprint(dp[n][0] % MOD)\n",
            6,
            debug=True,
        )
    )
