from utils import load_bigcodebench_hard, split_test_cases
import json
import argparse
from pathlib import Path
from collections import defaultdict

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_levels(level_dir):
    """
    Load level indices from the given directory.
    Returns a dict: task_id -> level ("Easy", "Medium", "Hard")
    """
    level_map = {}
    level_dir = Path(level_dir)
    
    for level_name in ["easy", "medium", "hard"]:
        file_path = level_dir / f"{level_name}.json"
        if file_path.exists():
            data = load_json(file_path)
            # data['ids'] should contain the list of task_ids
            for task_id in data.get('ids', []):
                level_map[task_id] = level_name.capitalize()
    
    return level_map

def main():
    parser = argparse.ArgumentParser(description="Calculate prediction accuracy against ground truth.")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--truth_file", type=str, required=True, help="Path to nucleus_eval_all.json")
    parser.add_argument("--level_dir", type=str, default=None, help="Directory containing level index files (easy.json, etc.)")
    
    args = parser.parse_args()
    
    print(f"Loading predictions from {args.pred_file}...")
    preds = load_jsonl(args.pred_file)
    pred_map_by_id = {item['id']: item for item in preds}

    print(f"Loading execution results (Ground Truth outcomes) from {args.truth_file}...")
    exec_data = load_json(args.truth_file)
    # Map: task_id -> metadata (execution logs)
    exec_map = {}
    for item in exec_data:
        t_id = item.get('question_id') or item.get('task_id')
        if t_id:
            exec_map[t_id] = item

    # Determine Level Directory
    if args.level_dir:
        level_dir = Path(args.level_dir)
    else:
        # Default assumption: ../actual_exec/problem_level_index relative to this script?
        base_dir = Path(__file__).resolve().parent.parent 
        level_dir = base_dir / "actual_exec" / "problem_level_index"

    print(f"Loading Level Index from {level_dir}...")
    task_level_map = load_levels(level_dir)

    print("Loading Dataset Registry (Total 854 TCs)...")
    dataset = load_bigcodebench_hard()
    
    total_tasks = 0
    correct_tasks = 0
    
    total_tcs = 0
    correct_tcs = 0
    
    missing_tasks_in_pred = 0
    
    # Level-based aggregations
    level_stats = defaultdict(lambda: {"total_tasks": 0, "correct_tasks": 0, "total_tcs": 0, "correct_tcs": 0})

    results = []
    
    # Iterate over EVERY task in the benchmark
    for item in dataset:
        task_id = item.get('task_id')
        test_code = item.get('test', '')
        level = task_level_map.get(task_id, "Unknown")
        
        total_tasks += 1
        level_stats[level]["total_tasks"] += 1
        
        # 1. Determine Actual Outcome (PASS/FAIL) from Execution
        # If task failed to run (missing in exec_map), it's a FAIL.
        exec_item = exec_map.get(task_id)
        
        # We need the specific code index results. 
        # Usually prediction is for code_index 0 (first sample).
        target_code_idx = 0 
        
        # Check Task Level Pass/Fail (Greedy/First sample)
        actual_task_pass = False
        actual_tc_outcomes = {} # {test_name: "PASS"|"FAIL"}
        
        # Identify all expected test cases from parsing
        split = split_test_cases(test_code)
        expected_tcs = [t[0] for t in split if t[0] not in ("error_parsing", "no_class_found", "no_test_methods")]
        expected_tcs = sorted(list(set(expected_tcs))) # Deduplicate and sort
        
        # Default all to FAIL first
        for tc in expected_tcs:
            actual_tc_outcomes[tc] = {"status": "FAIL", "pred": "NULL", "correct": False}
            
        if exec_item:
            graded_list = exec_item.get('graded_list', [])
            if target_code_idx < len(graded_list):
                actual_task_pass = graded_list[target_code_idx]
            
            # Detailed TC outcomes
            metadata_list = exec_item.get('metadata', [])
            if target_code_idx < len(metadata_list):
                meta = metadata_list[target_code_idx]
                if meta:
                    times_map = meta.get('times', {})
                    failures_map = meta.get('failures', {})
                    
                    for tc in expected_tcs:
                        # Check signatures
                        ran = False
                        failed = False
                        
                        for key in times_map.keys():
                            if key.endswith(f".{tc}") or key == tc:
                                ran = True
                                break
                        
                        for key in failures_map.keys():
                            if key.endswith(f".{tc}") or key == tc:
                                failed = True
                                break
                                
                        if ran and not failed:
                            actual_tc_outcomes[tc]["status"] = "PASS"
        
        # 2. Compare with Prediction
        pred_item = pred_map_by_id.get(task_id)
        
        # Task Level Comparison
        pred_task_pass = False
        if pred_item:
             pred_task_pass = (pred_item.get('overall_pass') == "PASS")
        else:
             missing_tasks_in_pred += 1
        
        if pred_task_pass == actual_task_pass:
            correct_tasks += 1
            level_stats[level]["correct_tasks"] += 1
            
        # TC Level Comparison
        pred_tc_list = pred_item.get('pass_fail_list', {}) if pred_item else {}
        
        for tc in expected_tcs:
            total_tcs += 1
            level_stats[level]["total_tcs"] += 1
            
            actual_status = actual_tc_outcomes[tc]["status"]
            pred_status = pred_tc_list.get(tc, "NULL")
            
            if pred_status == actual_status:
                correct_tcs += 1
                level_stats[level]["correct_tcs"] += 1
            
            # Update detailed log info
            actual_tc_outcomes[tc]["pred"] = pred_status
            actual_tc_outcomes[tc]["correct"] = (pred_status == actual_status)

        results.append({
            "task_id": task_id,
            "level": level,
            "overall_correct": pred_task_pass == actual_task_pass,
            "gt_overall": "PASS" if actual_task_pass else "FAIL",
            "pred_overall": "PASS" if pred_task_pass else "FAIL",
            "test_cases": actual_tc_outcomes
        })

    # Report
    task_acc = (correct_tasks / total_tasks * 100) if total_tasks else 0
    tc_acc = (correct_tcs / total_tcs * 100) if total_tcs else 0
    
    report_lines = [
        "=" * 60,
        "      ACCURACY REPORT (Ground Truth)      ",
        "=" * 60,
        f"{'Metric':<25} | {'Total':<10} | {'Correct':<10} | {'Accuracy':<10}",
        "-" * 60,
        f"{'Overall Task':<25} | {total_tasks:<10} | {correct_tasks:<10} | {task_acc:.2f}%",
        f"{'Overall Test Case':<25} | {total_tcs:<10} | {correct_tcs:<10} | {tc_acc:.2f}%",
        "-" * 60,
    ]

    # Append Level Reports
    for level in ["Easy", "Medium", "Hard", "Unknown"]:
        stats = level_stats.get(level)
        if not stats: continue
        
        l_total_tasks = stats["total_tasks"]
        l_correct_tasks = stats["correct_tasks"]
        l_task_acc = (l_correct_tasks / l_total_tasks * 100) if l_total_tasks else 0
        
        l_total_tcs = stats["total_tcs"]
        l_correct_tcs = stats["correct_tcs"]
        l_tc_acc = (l_correct_tcs / l_total_tcs * 100) if l_total_tcs else 0
        
        report_lines.append(f"{f'[{level}] Task':<25} | {l_total_tasks:<10} | {l_correct_tasks:<10} | {l_task_acc:.2f}%")
        report_lines.append(f"{f'[{level}] Test Case':<25} | {l_total_tcs:<10} | {l_correct_tcs:<10} | {l_tc_acc:.2f}%")
        report_lines.append("-" * 60)

    report_lines.append(f"Missing Predictions:         {missing_tasks_in_pred}")
    report_lines.append("=" * 60)
    
    report = "\n".join(report_lines)
    print(report)
    
    output_dir = Path(args.pred_file).parent
    
    # Save Report
    output_path = output_dir / "accuracy_report.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved to: {output_path}")

    # Save Raw Debug Log
    raw_path = output_dir / "accuracy_raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print(f"Detailed raw logs saved to: {raw_path}")

if __name__ == "__main__":
    main()
