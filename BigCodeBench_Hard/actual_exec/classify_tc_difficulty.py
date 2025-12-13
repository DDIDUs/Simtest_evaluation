import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import collections

# Paths
RESULTS_DIR = Path("/Users/woo/Documents/Simtest_evaluation/BigCodeBench_Hard/actual_exec/results/qwen3-coder-30B-A3B-instruct")
INPUT_FILE = RESULTS_DIR / "nucleus_eval_all.json"
OUTPUT_DIR = Path("/Users/woo/Documents/Simtest_evaluation/BigCodeBench_Hard/actual_exec/tc_level_index")

def main():
    if not INPUT_FILE.exists():
        print(f"Error: Input file found at {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # Dictionary to store pass/total counts for each test case
    # Key: "ProblemID#TestCaseName", Value: [pass_count, total_count]
    tc_stats = collections.defaultdict(lambda: [0, 0])

    for item in data:
        question_id = item.get("question_id", "Unknown")
        metadata_list = item.get("metadata", [])
        
        # We assume code_list size maps 1:1 to metadata_list
        # But we only care about aggregating stats across all samples
        
        for meta in metadata_list:
            if not meta:
                continue
                
            times_map = meta.get("times", {})
            failures_map = meta.get("failures", {})
            
            # Identify all executed test cases from 'times'
            # Key format usually: "candidate.TestCases.test_case_1"
            for complex_key in times_map.keys():
                # Normalize key: "test_case_1"
                simple_key = complex_key.split('.')[-1]
                unique_id = f"{question_id}#{simple_key}"
                
                # Check pass/fail
                # Failed if complex_key OR simple_key is in failures
                is_fail = (complex_key in failures_map) or (simple_key in failures_map)
                
                tc_stats[unique_id][1] += 1 # Increment total
                if not is_fail:
                    tc_stats[unique_id][0] += 1 # Increment pass

    # Classify
    easy_tcs = []
    medium_tcs = []
    hard_tcs = []
    all_pass_rates = []

    for unique_id, (pass_count, total) in tc_stats.items():
        if total == 0:
            continue
            
        rate = pass_count / total
        all_pass_rates.append(rate)
        
        if rate == 1.0:
            easy_tcs.append(unique_id)
        elif rate == 0.0:
            hard_tcs.append(unique_id)
        else:
            medium_tcs.append(unique_id)

    # Create Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(all_pass_rates, bins=11, range=(-0.05, 1.05), alpha=0.6, color='lightgreen', edgecolor='black', label='Test Case Count')
    plt.title('Test Case Pass Rate Distribution (0.0 - 1.0)')
    plt.xlabel('Pass Rate')
    plt.ylabel('Number of Test Cases')
    plt.xticks(np.arange(0, 1.1, 0.1))
    
    # 1. User specified thresholds (0-10%, 90-100%)
    plt.axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='User Threshold (10%, 90%)')
    plt.axvline(x=0.9, color='red', linestyle='--', linewidth=2)
    
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plot_path = OUTPUT_DIR / "tc_pass_rate_distribution.png"
    plt.savefig(plot_path)
    print(f"Distribution plot saved to: {plot_path}")

    # Save JSONs
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

    print(f"TC Level Classification Complete:")
    print(f"Easy (100% pass): {len(easy_tcs)}")
    print(f"Medium (Mixed):   {len(medium_tcs)}")
    print(f"Hard (0% pass):   {len(hard_tcs)}")
    print(f"Total Test Cases: {len(all_pass_rates)}")
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
