# run_lcb_eval.py

import argparse
import json

from compute_code_generation_metrics import evaluate_generations
from datasets import load_dataset

import pickle
import zlib
import base64


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
    """
    입력 JSON의 question_id를 기준으로:
      - LCB 데이터셋에서 같은 question_id 가진 row → 테스트 샘플 생성
      - JSON의 code_list → generations_by_qid[qid] 저장
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # LiveCodeBench 테스트셋 로드
    lcb_codegen = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag="release_latest",
    )

    # LCB row를 qid로 빠르게 찾기 위한 dict
    lcb_by_qid = {row["question_id"]: row for row in lcb_codegen}

    samples_list = []
    generations_by_qid = {}

    # JSON 기준으로 qid 매칭
    for item in data:
        qid = str(item["question_id"])
        code_list = item["code_list"]

        if qid not in lcb_by_qid:
            raise ValueError(f"LiveCodeBench 데이터셋에 question_id {qid} 없음")

        row = lcb_by_qid[qid]
        sample = build_evaluation_sample(row)

        # safety
        sample["question_id"] = qid

        samples_list.append(sample)
        generations_by_qid[qid] = code_list

    return samples_list, generations_by_qid, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--num_process_evaluate", type=int, default=16)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # JSON 로드: samples_list + generations_by_qid + 원본 data
    samples_list, generations_by_qid, data = load_samples_from_json(args.input_json)

    # 평가 수행 (qid 기반)
    results, metadatas = evaluate_generations(
        samples_list=samples_list,
        generations_by_qid=generations_by_qid,
        debug=args.debug,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
    )

    # 평가 결과를 원본 JSON에 병합
    for item in data:
        qid = str(item["question_id"])
        item["eval_result"] = results.get(qid)
        item["eval_metadata"] = metadatas.get(qid)

    # 저장
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
