import json
import os
from pathlib import Path

INPUT_FILE = Path("/Users/woo/Documents/Simtest_evaluation/BigCodeBench_Hard/actual_exec/results/qwen3-coder-30B-A3B-instruct/nucleus_eval_all.json")

def main():
    if not INPUT_FILE.exists():
        print(f"Error: Input file NOT found at {INPUT_FILE}")
        return

    print("Loading JSON...")
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items.")

    count = 0
    total_failure_ids = []

    for item in data:
        qid = item["question_id"]
        metadata_list = item.get("metadata", [])
        
        # We need ALL 10 samples to be 100% failures.
        # Check if we even have metadata
        if not metadata_list:
            continue

        all_samples_totally_failed = True
        
        for idx, meta in enumerate(metadata_list):
            failures = meta.get("failures", {})
            times = meta.get("times", {})
            
            # Condition for total failure of THIS sample:
            # All executed tests must be in failures list.
            # i.e., len(failures) == len(times)
            # Edge case: len(times) == 0 (no tests run? syntax error?). Usually syntax error counts as failure.
            
            # If ANY test passed (times > failures), then this sample is NOT a total failure.
            if len(times) > len(failures):
                all_samples_totally_failed = False
                break
        
        if all_samples_totally_failed:
            count += 1
            total_failure_ids.append(qid)

    print(f"Total problems where ALL samples failed ALL tests: {count}")
    print(f"IDs (first 10): {total_failure_ids[:10]}")

if __name__ == "__main__":
    main()
