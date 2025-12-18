import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

def load_tc_pass_rate_data(filepath):
    """
    Loads TC pass rate from count.json.
    Returns: dict { 'task_id#tc_name': float_pass_rate }
    """
    if not os.path.exists(filepath):
        print(f"Error: Count file not found at {filepath}")
        return {}
        
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    tc_pass_rate_map = {}
    for key, value in data.items():
        count = value.get('count', 0)
        pass_rate = count / 10.0
        tc_pass_rate_map[key] = pass_rate
    return tc_pass_rate_map

def load_task_pass_rate_data(filepath):
    """
    Loads Task pass rate from nucleus_eval_all.json (Qwen Data).
    Returns: dict { 'task_id': float_pass_rate }
    """
    if not os.path.exists(filepath):
        print(f"Error: Task eval file not found at {filepath}")
        return {}

    with open(filepath, 'r') as f:
        data = json.load(f)
        
    task_pass_rate_map = {}
    for item in data:
        task_id = item.get("question_id")
        graded = item.get("graded_list", [])
        # Pass rate logic
        p_rate = sum(graded) / len(graded) if graded else 0.0
        task_pass_rate_map[task_id] = p_rate
        
    return task_pass_rate_map

def load_accuracy_data(filepath, tc_map, task_map):
    """
    Joins accuracy data with TC pass rate and Task pass rate.
    Returns list of dicts.
    """
    if not os.path.exists(filepath):
        print(f"Error: Accuracy file not found at {filepath}")
        return []

    results = []
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            task_id = entry['task_id']
            test_cases = entry.get('test_cases', {})
            
            task_pass_rate = task_map.get(task_id)
            if task_pass_rate is None:
                continue

            for tc_name, tc_data in test_cases.items():
                tc_key = f"{task_id}#{tc_name}"
                tc_pass_rate = tc_map.get(tc_key)
                
                if tc_pass_rate is not None:
                    correct = tc_data['correct']
                    results.append({
                        'tc_pass_rate': tc_pass_rate,
                        'task_pass_rate': task_pass_rate,
                        'correct': correct
                    })
    return results

def main():
    # Argument Parsing
    import argparse
    parser = argparse.ArgumentParser(description="Generate Heatmap for a specific model.")
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt-5-mini-2025-08-07)")
    args = parser.parse_args()
    model_name = args.model

    # Dynamic Path Resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Script is in .../BigCodeBench_Hard/1_pred/heatmap.py
    
    # Root Dir resolution (assuming script is in BigCodeBench_Hard/1_pred)
    # We want to find the root of the repo to locate actual_exec/tc_level_index/count.json
    # Path: .../BigCodeBench_Hard/1_pred/heatmap.py
    # Root: .../
    
    path_parts = script_dir.split(os.sep)
    try:
        idx = path_parts.index("BigCodeBench_Hard")
        root_dir = os.sep.join(path_parts[:idx])
    except ValueError:
        # Fallback if structure is different, assume script is deep enough
        root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    # Fixed Paths (Standard of Truth)
    count_json_path = os.path.join(root_dir, "BigCodeBench_Hard/actual_exec/tc_level_index/count.json")
    # ALWAYS use Qwen's eval data for Task Difficulty
    task_eval_path = os.path.join(root_dir, "BigCodeBench_Hard/actual_exec/results/qwen3-coder-30B-A3B-instruct/nucleus_eval_all.json")
    
    # Model Specific Data
    # Input: results/{model_name}/accuracy_raw.jsonl
    model_dir = os.path.join(script_dir, "results", model_name)
    accuracy_jsonl_path = os.path.join(model_dir, "accuracy_raw.jsonl")

    # Output: results/{model_name}/correlation/
    output_dir = os.path.join(model_dir, "correlation")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Root Dir: {root_dir}")
    print(f"Model Dir: {model_dir}")
    print(f"Output Dir: {output_dir}")

    # Load mappings
    tc_pass_map = load_tc_pass_rate_data(count_json_path)
    task_pass_map = load_task_pass_rate_data(task_eval_path)
    if not tc_pass_map or not task_pass_map:
        return

    # Load and join data
    data = load_accuracy_data(accuracy_jsonl_path, tc_pass_map, task_pass_map)
    
    if not data:
        print("No matching accuracy data found!")
        return

    df = pd.DataFrame(data)
    
    # Binning Config
    bins = [i/10.0 for i in range(11)]
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    
    # Create Bins
    df['tc_bin'] = pd.cut(df['tc_pass_rate'], bins=bins, labels=labels, include_lowest=True, right=True)
    df['task_bin'] = pd.cut(df['task_pass_rate'], bins=bins, labels=labels, include_lowest=True, right=True)
    
    # Group by both bins
    heatmap_stats = df.groupby(['task_bin', 'tc_bin'], observed=False)['correct'].mean().reset_index()
    heatmap_counts = df.groupby(['task_bin', 'tc_bin'], observed=False)['correct'].count().reset_index()
    heatmap_sum = df.groupby(['task_bin', 'tc_bin'], observed=False)['correct'].sum().reset_index() # Count of True (1)
    
    # Pivot for Heatmap (Y=Task, X=TC)
    pivot_accuracy = heatmap_stats.pivot(index='task_bin', columns='tc_bin', values='correct')
    pivot_total = heatmap_counts.pivot(index='task_bin', columns='tc_bin', values='correct')
    pivot_correct = heatmap_sum.pivot(index='task_bin', columns='tc_bin', values='correct')
    
    # Create Annotation Matrix
    # Format: "75.0%\n(3/4)"
    annotations = pd.DataFrame(index=pivot_accuracy.index, columns=pivot_accuracy.columns)
    
    for r in pivot_accuracy.index:
        for c in pivot_accuracy.columns:
            acc = pivot_accuracy.loc[r, c]
            total = pivot_total.loc[r, c]
            correct = pivot_correct.loc[r, c]
            
            if pd.isna(acc) or total == 0:
                annotations.loc[r, c] = ""
            else:
                annotations.loc[r, c] = f"{acc:.1%}\n({int(correct)}/{int(total)})"

    # Save Stats
    stats_path = os.path.join(output_dir, "heatmap_2d_stats.json")
    heatmap_stats['count'] = heatmap_counts['correct']
    heatmap_stats['correct_count'] = heatmap_sum['correct']
    heatmap_stats.rename(columns={'correct': 'average_accuracy'}, inplace=True)
    heatmap_stats.to_json(stats_path, orient='records', indent=2)
    print(f"Stats saved to {stats_path}")

    # Determine Method Name
    parent_dir = os.path.basename(script_dir)
    method_map = {
        "1_pred": "Näive",
        "2_bug_local": "Diagnostic",
        "3_bug_report": "Rationale-Guided"
    }
    method_name = method_map.get(parent_dir, parent_dir)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    ax = sns.heatmap(pivot_accuracy, annot=annotations, fmt="", cmap="RdYlGn", vmin=0, vmax=1, 
                     cbar_kws={'label': 'Average Accuracy'})
    
    # Invert Y axis to have 0.0 at bottom
    ax.invert_yaxis()
    
    plt.title(f"LLM Accuracy Heatmap ({model_name} - {method_name})\n(X: Test Case Pass Rate, Y: Task Pass Rate)")
    plt.xlabel("Test Case Pass Rate (0.0=Hardest -> 1.0=Easiest)")
    plt.ylabel("Task Pass Rate (0.0=Hardest -> 1.0=Easiest)")
    
    output_img_path = os.path.join(output_dir, "heatmap_2d.png")
    plt.savefig(output_img_path, bbox_inches='tight')
    print(f"Heatmap saved to {output_img_path}")

if __name__ == "__main__":
    main()
