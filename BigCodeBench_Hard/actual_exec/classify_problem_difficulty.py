import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Paths
RESULTS_DIR = Path("/home/yrwoo/ICST26/Simtest_evaluation/BigCodeBench_Hard/actual_exec/results/qwen3-coder-30B-A3B-instruct")
INPUT_FILE = RESULTS_DIR / "nucleus_eval_all.json"
OUTPUT_DIR = Path("/home/yrwoo/ICST26/Simtest_evaluation/BigCodeBench_Hard/actual_exec/problem_level_index")

def main():
    if not INPUT_FILE.exists():
        print(f"Error: Input file found at {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    easy_tasks = []
    medium_tasks = []
    hard_tasks = []

    # Collect pass rates for plotting
    all_pass_rates = []
    
    for item in data:
        graded = item.get("graded_list", [])
        p_rate = sum(graded) / len(graded) if graded else 0.0
        all_pass_rates.append(p_rate)
        
        task_id = item.get("question_id")
        
        # User defined logic (Exact 0 and Exact 1)
        if p_rate == 1.0:
            easy_tasks.append(task_id)
        elif p_rate == 0.0:
            hard_tasks.append(task_id)
        else:
            medium_tasks.append(task_id)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(all_pass_rates, bins=11, range=(-0.05, 1.05), alpha=0.6, color='skyblue', edgecolor='black', label='Task Count')
    plt.title('Pass Rate Distribution (0.0 - 1.0)')
    plt.xlabel('Pass Rate')
    plt.ylabel('Number of Tasks')
    plt.xticks(np.arange(0, 1.1, 0.1))
    
    # 1. User specified thresholds (0-10%, 90-100%)
    # Interpret 0-10% as <= 0.1, 90-100% as >= 0.9
    plt.axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='User Threshold (10%, 90%)')
    plt.axvline(x=0.9, color='red', linestyle='--', linewidth=2)
    
    # 2. Distribution-based 3-split (Quantiles: 33%, 66%)
    q1 = np.percentile(all_pass_rates, 33.3)
    q2 = np.percentile(all_pass_rates, 66.6)
    
    plt.axvline(x=q1, color='green', linestyle='-.', linewidth=2, label=f'Tertile Split (33%={q1:.2f}, 66%={q2:.2f})')
    # Only plot q2 if distinct
    if q2 != q1:
         plt.axvline(x=q2, color='green', linestyle='-.', linewidth=2)
         
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plot_path = OUTPUT_DIR / "pass_rate_distribution_analysis.png"
    # Ensure directory exists before saving plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(plot_path)
    print(f"Distribution plot saved to: {plot_path}")

    # Save results (User logic)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def save_category(name, task_list):
        path = OUTPUT_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump({
                "count": len(task_list),
                "ids": task_list
            }, f, indent=2)

    save_category("easy", easy_tasks)
    save_category("medium", medium_tasks)
    save_category("hard", hard_tasks)

    print(f"Classification Complete (User Logic: 0/Mixed/100):")
    print(f"Easy (100% pass, >0.9): {len(easy_tasks)}")
    print(f"Medium (Mixed, 0.1-0.9): {len(medium_tasks)}")
    print(f"Hard (0% pass, <0.1):   {len(hard_tasks)}")
    print(f"Saved to {OUTPUT_DIR}")
    
    print("-" * 30)
    print(f"Distribution-based 3-split (Quantiles):")
    print(f"Lower 33%: Pass Rate <= {q1:.2f}")
    print(f"Middle 33%: {q1:.2f} < Pass Rate <= {q2:.2f}")
    print(f"Upper 33%: Pass Rate > {q2:.2f}")

if __name__ == "__main__":
    main()
