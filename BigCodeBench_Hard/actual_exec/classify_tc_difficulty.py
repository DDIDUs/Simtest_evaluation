import json
import os
import collections
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import load_bigcodebench_hard, split_test_cases

# Paths
RESULTS_DIR = Path("/home/yrwoo/ICST26/Simtest_evaluation/BigCodeBench_Hard/actual_exec/results/qwen3-coder-30B-A3B-instruct")
INPUT_FILE = RESULTS_DIR / "nucleus_eval_all.json"
OUTPUT_DIR = Path("/home/yrwoo/ICST26/Simtest_evaluation/BigCodeBench_Hard/actual_exec/tc_level_index")

PASS_STR = "pass"
MAX_CODE_IDX = 10  # code_idx: 0~9

def compute_passed_testcases_from_meta(meta: dict) -> set:
    """
    Return a set of testcase method names that are considered PASS for this code sample.
    Heuristic: passed = appeared in time_breakdown AND not in details(failures/errors)
    """
    if not meta:
        return set()

    # Remove strict status check. 
    # Even if status is "fail" (some tests failed), other tests might have passed.
    # We rely on time_breakdown presence to know what ran.
    # if meta.get("status") != PASS_STR:
    #    return set()

    times_map = meta.get("time_breakdown", {}) or {}
    failures_map = meta.get("details", {}) or {}

    passed = set()
    for complex_key in times_map.keys():
        simple_key = complex_key.split(".")[-1]
        is_fail = (complex_key in failures_map) or (simple_key in failures_map)
        if not is_fail:
            passed.add(simple_key)

    return passed


def main():
    print("Starting classification + heatmap script...", flush=True)
    if not INPUT_FILE.exists():
        print(f"Error: Input file NOT found at {INPUT_FILE}", flush=True)
        return

    # 1) Load GT and build global testcase registry
    print("Loading Ground Truth from BigCodeBench-Hard...", flush=True)
    gt_data = load_bigcodebench_hard()

    # Map: task_id -> list of test case names (method names)
    task_tc_map = collections.defaultdict(list)

    # Stats: unique_id -> {pass, total}
    tc_stats = {}

    # Global ordered list of testcase_ids
    global_tc_ids = []
    total_gt_tcs = 0

    print("Parsing Ground Truth Test Cases...", flush=True)
    for item in tqdm(gt_data, desc="Parsing GT"):
        t_id = item.get("task_id")
        test_code = item.get("test", "")
        split = split_test_cases(test_code)

        for method_name, _ in split:
            if method_name in ("error_parsing", "no_class_found", "no_test_methods"):
                continue

            task_tc_map[t_id].append(method_name)
            unique_id = f"{t_id}#{method_name}"

            tc_stats[unique_id] = {"pass": 0, "total": 0, "code_index": []}
            global_tc_ids.append(unique_id)
            total_gt_tcs += 1

    print(f"Initialized registry with {len(task_tc_map)} tasks and {total_gt_tcs} global unique testcases.", flush=True)

    # 2) Load execution results
    print(f"Loading Execution Results from {INPUT_FILE}...", flush=True)
    with open(INPUT_FILE, "r") as f:
        exec_data = json.load(f)

    # Build: task_id -> metadata_list (len = num_codes)
    task_meta_map = {}
    print("Indexing execution metadata by task...", flush=True)
    for item in tqdm(exec_data, desc="Indexing Exec"):
        task_id = item.get("question_id") or item.get("task_id")
        if not task_id:
            continue
        meta_list = item.get("metadata", []) or []
        task_meta_map[task_id] = meta_list

    # 3) Existing: Update tc_stats (pass/total) from execution results
    # (This preserves your previous easy/medium/hard classification logic.)
    print("Processing execution data for tc_stats...", flush=True)
    processed_tasks = 0
    for task_id, known_tcs in tqdm(task_tc_map.items(), desc="Processing tc_stats"):
        meta_list = task_meta_map.get(task_id, [])

        # For each generated code sample (metadata index)
        for idx, meta in enumerate(meta_list):
            # counts as an attempt for all known testcases
            for tc_name in known_tcs:
                tc_stats[f"{task_id}#{tc_name}"]["total"] += 1

            passed_in_sample = compute_passed_testcases_from_meta(meta)
            for tc_name in known_tcs:
                if tc_name in passed_in_sample:
                    tc_stats[f"{task_id}#{tc_name}"]["pass"] += 1
                    tc_stats[f"{task_id}#{tc_name}"]["code_index"].append(idx)

        processed_tasks += 1

    print(f"Processed tc_stats for {processed_tasks} tasks.", flush=True)

    # Heatmap removed as per user request
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5) Existing: classify and save histogram + jsons
    easy_tcs = []
    medium_tcs = []
    hard_tcs = []
    all_pass_rates = []

    # Pass Rate Bucket Counts: 0/10 (0%), 1/10 (10%), ..., 10/10 (100%)
    bucket_counts = [0] * 11  # Indices 0 to 10

    for unique_id, stats in tc_stats.items():
        pass_count = stats["pass"]
        total = stats["total"]
        
        rate = 0.0 if total == 0 else pass_count / total
        all_pass_rates.append(rate)

        # Classify Easy/Medium/Hard
        if rate == 1.0:
            easy_tcs.append(unique_id)
        elif rate == 0.0:
            hard_tcs.append(unique_id)
        else:
            medium_tcs.append(unique_id)

        # Bucket count for plotting
        # If total=10, pass_count is 0..10 -> direct index
        # If total != 10 (rarely), map to nearest bucket
        if total > 0:
             bucket_idx = int(round(rate * 10))
             bucket_counts[bucket_idx] += 1
        else:
             bucket_counts[0] += 1 # 0% pass

    # Bar Chart
    plt.figure(figsize=(10, 6))
    x_indices = range(11) # 0 to 10
    bars = plt.bar(x_indices, bucket_counts, color='skyblue', edgecolor='black')
    
    plt.title(f'Test Case Pass Rate Distribution (N={len(all_pass_rates)})')
    plt.xlabel('Pass Rate')
    plt.ylabel('Number of Test Cases')
    
    # X-axis labels: 0%, 10%, ..., 100%
    xtick_labels = [f"{i*10}%" for i in x_indices]
    plt.xticks(x_indices, xtick_labels)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')
    
    plt.grid(axis='y', alpha=0.3)
    
    plot_path = OUTPUT_DIR / "tc_pass_rate_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Distribution plot saved to: {plot_path}", flush=True)

    def save_category(name, tc_list):
        path = OUTPUT_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump({"count": len(tc_list), "ids": tc_list}, f, indent=2)

    save_category("easy", easy_tcs)
    save_category("medium", medium_tcs)
    save_category("hard", hard_tcs)

    # Save detailed counts
    count_data = {}
    for unique_id, stats in tc_stats.items():
        count_data[unique_id] = {
            "count": stats["pass"],
            "code_index": stats["code_index"]
        }
    
    with open(OUTPUT_DIR / "count.json", "w") as f:
        json.dump(count_data, f, indent=2)

    print("-" * 30, flush=True)
    print("TC Level Classification Complete:", flush=True)
    print(f"Easy (100% pass): {len(easy_tcs)}", flush=True)
    print(f"Medium (Mixed):   {len(medium_tcs)}", flush=True)
    print(f"Hard (0% pass):   {len(hard_tcs)}", flush=True)
    print(f"Total Test Cases: {len(all_pass_rates)}", flush=True)
    print(f"Saved to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()