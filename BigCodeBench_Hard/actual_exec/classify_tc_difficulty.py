import json
import os
import collections
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import load_bigcodebench_hard, split_test_cases

# Paths
RESULTS_DIR = Path("/Users/woo/Documents/Simtest_evaluation/BigCodeBench_Hard/actual_exec/results/qwen3-coder-30B-A3B-instruct")
INPUT_FILE = RESULTS_DIR / "nucleus_eval_all.json"
OUTPUT_DIR = Path("/Users/woo/Documents/Simtest_evaluation/BigCodeBench_Hard/actual_exec/tc_level_index")

def main():
    print("Starting classification script...", flush=True)
    if not INPUT_FILE.exists():
        print(f"Error: Input file NOT found at {INPUT_FILE}", flush=True)
        return

    # 1. Build Ground Truth Registry
    print("Loading Ground Truth from BigCodeBench-Hard...", flush=True)
    gt_data = load_bigcodebench_hard()
    
    # Map: task_id -> list of test case names
    task_tc_map = collections.defaultdict(list)
    # Stats: unique_id -> {pass, total}
    tc_stats = {}
    
    total_gt_tcs = 0
    
    print("Parsing Ground Truth Test Cases...", flush=True)
    for item in tqdm(gt_data, desc="Parsing GT"):
        t_id = item.get("task_id")
        test_code = item.get("test", "")
        split = split_test_cases(test_code)
        
        for method_name, _ in split:
            # Skip invalid/error parsing results
            if method_name in ("error_parsing", "no_class_found", "no_test_methods"):
                continue
                
            task_tc_map[t_id].append(method_name)
            unique_id = f"{t_id}#{method_name}"
            tc_stats[unique_id] = {"pass": 0, "total": 0}
            total_gt_tcs += 1
            
    print(f"Initialized registry with {len(task_tc_map)} tasks and {total_gt_tcs} unique test cases (Ground Truth).", flush=True)

    # 2. Process Execution Results
    print(f"Loading Execution Results from {INPUT_FILE}...", flush=True)
    with open(INPUT_FILE, "r") as f:
        exec_data = json.load(f)

    print("Processing Execution Data...", flush=True)
    
    processed_count = 0
    
    for item in tqdm(exec_data, desc="Processing Exec"):
        # Resolve ID
        task_id = item.get("question_id") or item.get("task_id")
        if not task_id: continue
        
        # If this task didn't exist in GT (unlikely), skip
        if task_id not in task_tc_map:
            # print(f"Warning: Task {task_id} found in Exec but not in GT.")
            continue
            
        metadata_list = item.get("metadata", [])
        
        # We need to know the 'known' test cases for this task
        known_tcs = task_tc_map[task_id]
        
        # For each sample generated
        for meta in metadata_list:
            # Increment TOTAL for all known test cases (since this sample 'attempted' them)
            # Even if meta is empty or crashed, it counts as an attempt that failed.
            for tc_name in known_tcs:
                tc_stats[f"{task_id}#{tc_name}"]["total"] += 1
            
            if not meta: 
                continue

            times_map = meta.get("times", {})
            failures_map = meta.get("failures", {})
            
            # Identify passed tests in this sample
            # Passed = In 'times' (ran) AND Not in 'failures'
            passed_in_sample = set()
            
            for complex_key in times_map.keys():
                simple_key = complex_key.split('.')[-1]
                
                is_fail = (complex_key in failures_map) or (simple_key in failures_map)
                
                if not is_fail:
                    passed_in_sample.add(simple_key)
            
            # Increment PASS for passed ones
            for tc_name in known_tcs:
                if tc_name in passed_in_sample:
                    tc_stats[f"{task_id}#{tc_name}"]["pass"] += 1
                    
        processed_count += 1

    print(f"Processed execution results for {processed_count} tasks.")

    # 3. Classify and Save
    easy_tcs = []
    medium_tcs = []
    hard_tcs = []
    all_pass_rates = []

    for unique_id, stats in tc_stats.items():
        pass_count = stats["pass"]
        total = stats["total"]
        
        if total == 0:
            # Did not run at all? Treat as Hard (0%)
            rate = 0.0
        else:
            rate = pass_count / total
            
        all_pass_rates.append(rate)
        
        if rate == 1.0:
            easy_tcs.append(unique_id)
        elif rate == 0.0:
            hard_tcs.append(unique_id)
        else:
            medium_tcs.append(unique_id)

    # Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(all_pass_rates, bins=11, range=(-0.05, 1.05), alpha=0.6, color='skyblue', edgecolor='black', label='Test Case Count')
    plt.title(f'Test Case Pass Rate Distribution (N={len(all_pass_rates)})')
    plt.xlabel('Pass Rate')
    plt.ylabel('Number of Test Cases')
    plt.xticks(np.arange(0, 1.1, 0.1))
    
    plt.axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='Thresholds (10%, 90%)')
    plt.axvline(x=0.9, color='red', linestyle='--', linewidth=2)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plot_path = OUTPUT_DIR / "tc_pass_rate_distribution.png"
    plt.savefig(plot_path)
    print(f"Distribution plot saved to: {plot_path}")

    # JSONs
    def save_category(name, tc_list):
        path = OUTPUT_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump({
                "count": len(tc_list),
                "ids": tc_list
            }, f, indent=2)

    save_category("easy", easy_tcs)
    save_category("medium", medium_tcs)
    save_category("hard", hard_tcs)

    print("-" * 30)
    print(f"TC Level Classification Complete:")
    print(f"Easy (100% pass): {len(easy_tcs)}")
    print(f"Medium (Mixed):   {len(medium_tcs)}")
    print(f"Hard (0% pass):   {len(hard_tcs)}")
    print(f"Total Test Cases: {len(all_pass_rates)}")
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
