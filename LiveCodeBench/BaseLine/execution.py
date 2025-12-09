# run_lcb_eval.py

import argparse
import json
from collections import defaultdict

from compute_code_generation_metrics import evaluate_generations  # 위 코드가 들어있는 모듈 이름으로 수정
from datasets import load_dataset

import json
import pickle
import zlib
import base64
from datasets import load_dataset


def decode_private_test_cases(s: str):
    """private_test_cases 문자열을 json 리스트로 디코딩."""
    if not s:
        return []
    return json.loads(
        pickle.loads(
            zlib.decompress(
                base64.b64decode(s.encode("utf-8"))
            )
        )
    )  # [{"input": ..., "output": ...}, ...]


def build_evaluation_sample(row):
    # public_test_cases: JSON 문자열 or 이미 리스트일 수 있음
    public_raw = row["public_test_cases"]
    if isinstance(public_raw, str):
        public_cases = json.loads(public_raw)
    else:
        public_cases = public_raw

    # private_test_cases: base64 + zlib + pickle + json
    private_cases = decode_private_test_cases(row["private_test_cases"])

    all_cases = public_cases + private_cases

    inputs = [t["input"] for t in all_cases]
    outputs = [t["output"] for t in all_cases]

    # metadata: 문자열일 수도 있고 dict일 수도 있음
    meta_raw = row["metadata"]
    metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
    fn_name = metadata.get("func_name", None)

    # 질문에서 준 get_evaluation_sample 포맷
    eval_sample = {
        "input_output": json.dumps(
            {
                "inputs": inputs,
                "outputs": outputs,
                "fn_name": fn_name,
            }
        ),
        "question_id": row.get("question_id", None),
    }
    return eval_sample

def load_samples_from_json(json_path: str):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples_list = []
    generations_list = []
    
    lcb_codegen = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag="release_latest",
    )
    
    for row in lcb_codegen:
        samples_list.append(build_evaluation_sample(row))
    for item in data:
        generations_list.append(item["code_list"])

    return samples_list, generations_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="문제 및 코드가 들어있는 입력 JSON 파일 경로",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="테스트 결과를 저장할 JSON 파일 경로",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=6,
        help="각 코드 샘플에 대한 테스트 타임아웃(초)",
    )
    parser.add_argument(
        "--num_process_evaluate",
        type=int,
        default=16,
        help="테스트 수행 시 사용할 프로세스 개수",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug 출력 여부",
    )

    args = parser.parse_args()

    # 1) JSON에서 samples / generations 생성
    samples_list, generations_list = load_samples_from_json(args.input_json)

    # 2) 기존 평가 함수 호출 (run_test 체인 사용)
    results, metadatas = evaluate_generations(
        samples_list,
        generations_list,
        debug=args.debug,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
    )

    # 3) JSON 직렬화 (dict key 를 문자열로)
    results_json = {str(k): v for k, v in results.items()}
    metadatas_json = {str(k): v for k, v in metadatas.items()}

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results_json,
                "metadata": metadatas_json,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
