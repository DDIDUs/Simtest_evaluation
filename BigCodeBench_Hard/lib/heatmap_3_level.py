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
    parser = argparse.ArgumentParser(description="Generate Heatmap (3 Levels) for a specific model.")
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt-5-mini-2025-08-07)")
    parser.add_argument("--dir", default="1_no_reasoning", help="Target directory (e.g., 1_no_reasoning)")
    args = parser.parse_args()
    model_name = args.model
    target_dir = args.dir

    # Dynamic Path Resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Script is in .../BigCodeBench_Hard/lib/heatmap_3_level.py
    
    # Root Dir resolution
    # Assumes script is in <Root>/BigCodeBench_Hard/lib
    path_parts = script_dir.split(os.sep)
    try:
        idx = path_parts.index("BigCodeBench_Hard")
        root_dir = os.sep.join(path_parts[:idx])
    except ValueError:
        # Fallback if structure is different
        root_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    # Fixed Paths (Standard of Truth)
    count_json_path = os.path.join(root_dir, "BigCodeBench_Hard/actual_exec/tc_level_index/count.json")
    # ALWAYS use Qwen's eval data for Task Difficulty
    task_eval_path = os.path.join(root_dir, "BigCodeBench_Hard/actual_exec/results/qwen3-coder-30B-A3B-instruct/nucleus_eval_all.json")
    
    # Model Specific Data
    # Input: BigCodeBench_Hard/{target_dir}/results/{model_name}/accuracy_raw.jsonl
    base_dir = os.path.join(root_dir, "BigCodeBench_Hard", target_dir)
    model_dir = os.path.join(base_dir, "results", model_name)
    accuracy_jsonl_path = os.path.join(model_dir, "accuracy_raw.jsonl")

    # Output: results/{model_name}/correlation/
    # We save output in the SAME directory as input results
    output_dir = os.path.join(model_dir, "correlation")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Root Dir: {root_dir}")
    print(f"Base Dir: {base_dir}")
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
    
    # Determine Method Name
    method_map = {
        "1_no_reasoning": "No Reasoning",
        "2_non_code_specific": "Non-Code Specific",
        "3_code_specific": "Code Specific"
    }
    method_name = method_map.get(target_dir, target_dir)

    # Configs (Target: Combo 2 Logic)
    
    # Task: Hard (0.0), Medium (0.0 < x < 1.0), Easy (1.0) -> Label as Low, Medium, High
    task_bins = [-0.1, 0.0001, 0.9999, 1.1]
    task_labels = ["Low (0.0)", "Medium (0.0 < x < 1.0)", "High (1.0)"]
    
    # TC: Hard (0.0-0.1), Medium (0.1-0.9), Easy (0.9-1.0) -> Label as Hard, Medium, Easy
    tc_bins = [0.0, 0.1, 0.9, 1.0]
    tc_labels = ["Hard (0.0-0.1)", "Medium (0.1-0.9)", "Easy (0.9-1.0)"]
    
    # Binning
    df['tc_bin'] = pd.cut(df['tc_pass_rate'], bins=tc_bins, labels=tc_labels, include_lowest=True, right=True)
    df['task_bin'] = pd.cut(df['task_pass_rate'], bins=task_bins, labels=task_labels, include_lowest=True, right=True)
    
    # Group
    heatmap_stats = df.groupby(['task_bin', 'tc_bin'], observed=False)['correct'].mean().reset_index()
    heatmap_counts = df.groupby(['task_bin', 'tc_bin'], observed=False)['correct'].count().reset_index()
    heatmap_sum = df.groupby(['task_bin', 'tc_bin'], observed=False)['correct'].sum().reset_index()

    # Pivot
    pivot_accuracy = heatmap_stats.pivot(index='task_bin', columns='tc_bin', values='correct')
    pivot_total = heatmap_counts.pivot(index='task_bin', columns='tc_bin', values='correct')
    pivot_correct = heatmap_sum.pivot(index='task_bin', columns='tc_bin', values='correct')

    # Reorder Columns (Easy -> Hard) as requested
    # Labels must match exactly defined above
    column_order = ["Easy (0.9-1.0)", "Medium (0.1-0.9)", "Hard (0.0-0.1)"]
    pivot_accuracy = pivot_accuracy.reindex(columns=column_order)
    pivot_total = pivot_total.reindex(columns=column_order)
    pivot_correct = pivot_correct.reindex(columns=column_order)

    # Annotations
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
    stats_path = os.path.join(output_dir, "heatmap_3_level_stats.json")
    heatmap_stats['count'] = heatmap_counts['correct']
    heatmap_stats['correct_count'] = heatmap_sum['correct']
    heatmap_stats.rename(columns={'correct': 'average_accuracy'}, inplace=True)
    heatmap_stats.to_json(stats_path, orient='records', indent=2)
    print(f"Stats saved to {stats_path}")

    # Plotting
    plt.figure(figsize=(12, 10))
    # sns.set(font_scale=1.2) # Optional global scale
    
    ax = sns.heatmap(pivot_accuracy, annot=annotations, fmt="", cmap="RdYlGn", vmin=0, vmax=1, 
                     cbar_kws={'label': 'Average Accuracy'}, annot_kws={"size": 18})
    
    # Colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(16)
    cbar.ax.tick_params(labelsize=14)

    ax.invert_yaxis()
    
    # plt.title(f"LLM Accuracy Heatmap ({model_name} - {method_name})\n(X: Testcase Difficulty, Y: Code Quality)")
    plt.xlabel("Testcase Difficulty", fontsize=18)
    plt.ylabel("Code Quality", fontsize=18)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    output_img_path = os.path.join(output_dir, "heatmap_3_level.png")
    plt.savefig(output_img_path, bbox_inches='tight')
    print(f"Heatmap saved to {output_img_path}")
    plt.close()

if __name__ == "__main__":
    main()
