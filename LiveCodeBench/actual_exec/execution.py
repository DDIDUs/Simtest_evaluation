import argparse
import json

from compute_code_generation_metrics import evaluate_generations
from datasets import load_dataset

import pickle
import zlib
import base64


def decode_private_test_cases(s: str):
    if not s:
        return []
    return json.loads(
        pickle.loads(
            zlib.decompress(
                base64.b64decode(s.encode("utf-8"))
            )
        )
    )


def build_evaluation_sample(row):
    public_raw = row["public_test_cases"]
    public_cases = json.loads(public_raw) if isinstance(public_raw, str) else public_raw

    private_cases = decode_private_test_cases(row["private_test_cases"])

    all_cases = public_cases + private_cases
    inputs = [t["input"] for t in all_cases]
    outputs = [t["output"] for t in all_cases]

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
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lcb_codegen = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag="release_latest",
    )

    lcb_by_qid = {str(row["question_id"]): row for row in lcb_codegen}

    samples_list = []
    generations_by_qid = {}

    for item in data:
        qid = str(item["question_id"])
        code_list = item["code_list"]

        if qid not in lcb_by_qid:
            raise ValueError(f"LiveCodeBench 데이터셋에 question_id {qid} 없음")

        row = lcb_by_qid[qid]
        sample = build_evaluation_sample(row)
        sample["question_id"] = qid  # safety

        samples_list.append(sample)
        generations_by_qid[qid] = code_list

    return samples_list, generations_by_qid, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default='data/Scenario.codegeneration_10_0.7.json')
    parser.add_argument("--output_json", type=str, default='results/out.json')
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--num_process_evaluate", type=int, default=16)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    samples_list, generations_by_qid, data = load_samples_from_json(args.input_json)

    # qid -> row index
    qid_to_idx = {str(item["question_id"]): i for i, item in enumerate(data)}

    # (선택) 시작 시점에 빈 결과로 output 파일 하나 만들어두기
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # 샘플 1개 끝나면 즉시 반영 후 저장
    for qid, eval_result, difficulty in evaluate_generations(
        samples_list=samples_list,
        generations_by_qid=generations_by_qid,
        debug=args.debug,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
        show_progress=True,
    ):
        idx = qid_to_idx[qid]

        data[idx]["eval_result"] = eval_result
        data[idx]["difficulty"] = difficulty

        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
