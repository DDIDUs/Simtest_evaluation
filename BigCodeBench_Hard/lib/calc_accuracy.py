import json
import argparse
import os
from pathlib import Path
from collections import defaultdict
from utils import load_bigcodebench_hard, split_test_cases

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

def load_tc_pass_rates(count_path):
    """
    Load TC pass rates from count.json.
    Returns: dict { 'task_id#tc_name': pass_rate }
    """
    if not os.path.exists(count_path):
        print(f"Error: Count file not found at {count_path}")
        return {}
        
    with open(count_path, 'r') as f:
        data = json.load(f)
    
    tc_pass_rate_map = {}
    for key, value in data.items():
        count = value.get('count', 0)
        pass_rate = count / 10.0
        tc_pass_rate_map[key] = pass_rate
    return tc_pass_rate_map

def get_tc_category(pass_rate):
    if pass_rate >= 0.999: # 1.0
        return "All-P"
    elif pass_rate <= 0.001: # 0.0
        return "All-F"
    else:
        return "Mix"

def main():
    parser = argparse.ArgumentParser(description="Calculate prediction accuracy against ground truth.")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--truth_file", type=str, required=True, help="Path to nucleus_eval_all.json")
    # level_dir is no longer needed but kept for backward compatibility if scripts call it
    parser.add_argument("--level_dir", type=str, default=None, help="[Deprecated] Directory containing level index files")
    
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

    # Determine count.json path
    # Script is likely at BigCodeBench_Hard/lib/calc_accuracy.py or similar
    # We assume valid structure relative to this script or handle it dynamically
    # Plan assumption: BigCodeBench_Hard/lib/calc_accuracy.py
    # count.json: BigCodeBench_Hard/actual_exec/tc_level_index/count.json
    
    base_dir = Path(__file__).resolve().parent.parent 
    count_json_path = base_dir / "actual_exec" / "tc_level_index" / "count.json"
    
    print(f"Loading TC Pass Rates from {count_json_path}...")
    tc_pass_map = load_tc_pass_rates(count_json_path)
    if not tc_pass_map:
        print("Warning: No pass rate data found. Categorization will fail.")

    print("Loading Dataset Registry (Total 854 TCs)...")
    dataset = load_bigcodebench_hard()
    
    total_tasks = 0
    correct_tasks = 0
    
    total_tcs = 0
    correct_tcs = 0
    
    missing_tasks_in_pred = 0
    
    # Category-based aggregations for Test Cases
    # Structure: { "All-P": {"total_tcs": 0, "correct_tcs": 0}, ... }
    cat_tc_stats = defaultdict(lambda: {"total_tcs": 0, "correct_tcs": 0})

    # Category-based aggregations for Tasks (Code Quality)
    # Structure: { "Fully Correct": {"total_tasks": 0, "correct_tasks": 0, "total_tcs": 0, "correct_tcs": 0}, ...}
    cat_code_stats = defaultdict(lambda: {"total_tasks": 0, "correct_tasks": 0, "total_tcs": 0, "correct_tcs": 0})

    results = []
    
    # Iterate over EVERY task in the benchmark
    for item in dataset:
        task_id = item.get('task_id')
        test_code = item.get('test', '')
        
        total_tasks += 1
        
        # 1. Determine Actual Outcome (PASS/FAIL) from Execution
        exec_item = exec_map.get(task_id)
        
        # We need the specific code index results. 
        target_code_idx = 0 
        
        actual_task_pass = False
        actual_tc_outcomes = {} # {test_name: "PASS"|"FAIL"}
        
        # Identify all expected test cases from parsing
        split = split_test_cases(test_code)
        expected_tcs = [t[0] for t in split if t[0] not in ("error_parsing", "no_class_found", "no_test_methods")]
        expected_tcs = sorted(list(set(expected_tcs))) # Deduplicate and sort
        
        # Default all to FAIL first
        for tc in expected_tcs:
            actual_tc_outcomes[tc] = "FAIL"
            
        if exec_item:
            graded_list = exec_item.get('graded_list', [])
            if target_code_idx < len(graded_list):
                actual_task_pass = graded_list[target_code_idx]
            
            # Detailed TC outcomes
            metadata_list = exec_item.get('metadata', [])
            if target_code_idx < len(metadata_list):
                meta = metadata_list[target_code_idx]
                if meta:
                    times_map = meta.get('time_breakdown', {})
                    failures_map = meta.get('details', {})
                    
                    for tc in expected_tcs:
                        # Check signatures
                        ran = False
                        failed = False
                        
                        # Naive substring check or split check
                        for key in times_map.keys():
                            if key.endswith(f".{tc}") or key == tc:
                                ran = True
                                break
                        
                        for key in failures_map.keys():
                            if key.endswith(f".{tc}") or key == tc:
                                failed = True
                                break
                                
                        if ran and not failed:
                            actual_tc_outcomes[tc] = "PASS"
        
        # Calculate Code Quality Category
        # Based on Actual TC Pass Rate for this task
        num_expected = len(expected_tcs)
        num_passed_actual = sum(1 for status in actual_tc_outcomes.values() if status == "PASS")
        
        if num_expected > 0:
            pass_ratio = num_passed_actual / num_expected
        else:
            pass_ratio = 0.0 # Should not happen typically
            
        if pass_ratio >= 0.999:
            code_category = "Fully Correct"
        elif pass_ratio <= 0.001:
            code_category = "Incorrect"
        else:
            code_category = "Partially Correct"

        cat_code_stats[code_category]["total_tasks"] += 1

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
            cat_code_stats[code_category]["correct_tasks"] += 1
            
        # TC Level Comparison
        pred_tc_list = pred_item.get('pass_fail_list', {}) if pred_item else {}
        
        for tc in expected_tcs:
            total_tcs += 1
            cat_code_stats[code_category]["total_tcs"] += 1
            
            # Determine Category
            unique_id = f"{task_id}#{tc}"
            pass_rate = tc_pass_map.get(unique_id, -1) # Default to -1 if unknown
            if pass_rate == -1:
                # Fallback or unknown
                category = "Unknown"
            else:
                category = get_tc_category(pass_rate)
            
            cat_tc_stats[category]["total_tcs"] += 1
            
            actual_status = actual_tc_outcomes[tc]
            pred_status = pred_tc_list.get(tc, "NULL")
            
            is_correct = (pred_status == actual_status)
            if is_correct:
                correct_tcs += 1
                cat_tc_stats[category]["correct_tcs"] += 1
                cat_code_stats[code_category]["correct_tcs"] += 1
                
            # Log detailed result (optional, sticking to requested changes)
            actual_tc_outcomes[tc] = {
                "status": actual_status, 
                "pred": pred_status, 
                "correct": is_correct,
                "category": category
            }

        results.append({
            "task_id": task_id,
            "overall_correct": pred_task_pass == actual_task_pass,
            "gt_overall": "PASS" if actual_task_pass else "FAIL",
            "pred_overall": "PASS" if pred_task_pass else "FAIL",
            "code_category": code_category,
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

    # Append TC Category Reports
    # Order: All-P, Mix, All-F
    for category in ["All-P", "Mix", "All-F"]:
        stats = cat_tc_stats.get(category)
        if not stats: 
            # Print row even if 0 to show structure
            l_total_tcs = 0
            l_correct_tcs = 0
            l_tc_acc = 0.0
        else:
            l_total_tcs = stats["total_tcs"]
            l_correct_tcs = stats["correct_tcs"]
            l_tc_acc = (l_correct_tcs / l_total_tcs * 100) if l_total_tcs else 0
        
        report_lines.append(f"{f'TC [{category}]':<25} | {l_total_tcs:<10} | {l_correct_tcs:<10} | {l_tc_acc:.2f}%")
    
    report_lines.append("-" * 60)
    
    # Append Code Category Reports
    # Order: Fully Correct, Partially Correct, Incorrect
    for category in ["Fully Correct", "Partially Correct", "Incorrect"]:
        stats = cat_code_stats.get(category)
        if not stats:
            stats = {"total_tasks": 0, "correct_tasks": 0, "total_tcs": 0, "correct_tcs": 0}
            
        c_total_tasks = stats["total_tasks"]
        c_correct_tasks = stats["correct_tasks"]
        c_task_acc = (c_correct_tasks / c_total_tasks * 100) if c_total_tasks else 0
        
        c_total_tcs = stats["total_tcs"]
        c_correct_tcs = stats["correct_tcs"]
        c_tc_acc = (c_correct_tcs / c_total_tcs * 100) if c_total_tcs else 0
        
        report_lines.append(f"{f'Code [{category}] Task':<25} | {c_total_tasks:<10} | {c_correct_tasks:<10} | {c_task_acc:.2f}%")
        report_lines.append(f"{f'Code [{category}] TC':<25} | {c_total_tcs:<10} | {c_correct_tcs:<10} | {c_tc_acc:.2f}%")
        report_lines.append("-" * 60)

    # Calculate Prediction Stats
    pred_task_pass_count = 0
    pred_task_fail_count = 0
    pred_tc_pass_count = 0
    pred_tc_fail_count = 0

    for res in results:
        # Task Level
        if res['pred_overall'] == 'PASS':
            pred_task_pass_count += 1
        else:
            pred_task_fail_count += 1
        
        # TC Level
        for tc_name, tc_res in res['test_cases'].items():
            if tc_res['pred'] == 'PASS':
                pred_tc_pass_count += 1
            else:
                pred_tc_fail_count += 1

    report_lines.append("Prediction Distribution:")
    report_lines.append(f"  Tasks: PASS {pred_task_pass_count}/{total_tasks} ({(pred_task_pass_count/total_tasks*100) if total_tasks else 0:.1f}%), FAIL {pred_task_fail_count}/{total_tasks} ({(pred_task_fail_count/total_tasks*100) if total_tasks else 0:.1f}%)")
    report_lines.append(f"  Test Cases: PASS {pred_tc_pass_count}/{total_tcs} ({(pred_tc_pass_count/total_tcs*100) if total_tcs else 0:.1f}%), FAIL {pred_tc_fail_count}/{total_tcs} ({(pred_tc_fail_count/total_tcs*100) if total_tcs else 0:.1f}%)")
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
