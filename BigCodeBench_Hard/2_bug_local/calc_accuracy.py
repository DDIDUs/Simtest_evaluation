import json
import argparse
from pathlib import Path

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

def main():
    parser = argparse.ArgumentParser(description="Calculate prediction accuracy against ground truth.")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--truth_file", type=str, required=True, help="Path to nucleus_eval_all.json")
    
    args = parser.parse_args()
    
    print(f"Loading predictions from {args.pred_file}...")
    preds = load_jsonl(args.pred_file)
    
    print(f"Loading ground truth from {args.truth_file}...")
    truth_data = load_json(args.truth_file)
    
    # Create lookup map for ground truth
    # nucleus_eval_all.json items have 'question_id'
    truth_map = {item['question_id']: item for item in truth_data}
    
    total = 0
    correct = 0
    missing = 0
    
    # Granular counters
    total_test_cases = 0
    correct_test_cases = 0
    
    results = []
    
    for pred in preds:
        t_id = pred['id']
        
        if t_id not in truth_map:
            # Maybe ID mismatch? BigCodeBench IDs usually match.
            # Only count if we have ground truth
            print(f"Warning: Task ID {t_id} not found in ground truth.")
            missing += 1
            continue
            
        truth_item = truth_map[t_id]
        
        # Get code index used in prediction
        code_idx = pred.get('code_index', 0)
        
        # --- Overall Check ---
        pred_status = pred.get('overall_pass')
        pred_bool = (pred_status == "PASS")
        
        graded_list = truth_item.get('graded_list', [])
        if code_idx >= len(graded_list):
            print(f"Warning: Code index {code_idx} out of range for task {t_id} (len {len(graded_list)})")
            missing += 1
            continue
            
        actual_bool = graded_list[code_idx]
        is_correct = (pred_bool == actual_bool)
        
        total += 1
        if is_correct:
            correct += 1
            
        # --- Granular Check ---
        # Prediction Map: { "test_case_1": "PASS", "test_case_2": "FAIL" ... }
        pred_map = pred.get('pass_fail_list', {})
        
        # Ground Truth Extraction
        # truth_item['metadata'] is a list of metadata for each sample
        metadata_list = truth_item.get('metadata', [])
        if code_idx < len(metadata_list):
            meta = metadata_list[code_idx]
            failures_map = meta.get('failures', {})
            times_map = meta.get('times', {})
            
            # Identify all executed test cases from 'times' keys
            # Key format usually: "candidate.TestCases.test_case_1" or "test_case_1"
            # We need to normalize to just "test_case_N" to match pred_map keys
            
            for complex_key in times_map.keys():
                # Extract simple name
                # e.g. "candidate.TestCases.test_case_1" -> "test_case_1"
                simple_key = complex_key.split('.')[-1]
                
                # Check if this test case exists in prediction
                if simple_key not in pred_map:
                    # Maybe prediction skipped it or format differs?
                    continue
                
                # Ground Truth for this specific test case
                # If key is in failures_map -> FAIL, else PASS
                # failures map might use complex key OR simple key depending on implementation
                # Check both just in case
                is_fail = (complex_key in failures_map) or (simple_key in failures_map)
                gt_status = "FAIL" if is_fail else "PASS"
                
                # Prediction for this specific test case
                # pred_map values are "PASS", "FAIL", "NULL"
                p_status = pred_map[simple_key]
                
                total_test_cases += 1
                if p_status == gt_status:
                    correct_test_cases += 1
                
        else:
             print(f"Warning: Metadata missing for code index {code_idx} in task {t_id}")

        results.append({
            "id": t_id,

            "code_index": code_idx,
            "prediction": pred_status,
            "ground_truth": "PASS" if actual_bool else "FAIL",
            "is_correct": is_correct
        })
        
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    detail_accuracy = (correct_test_cases / total_test_cases) * 100 if total_test_cases > 0 else 0.0
    
    report_lines = [
        "-" * 40,
        f"Total Tasks Evaluated:       {total}",
        f"Correct Tasks (Overall):     {correct}",
        f"Task Accuracy (Overall):     {accuracy:.2f}%",
        "-" * 40,
        f"Total Test Cases Evaluated:  {total_test_cases}",
        f"Correct Test Cases:          {correct_test_cases}",
        f"Test Case Accuracy (Detail): {detail_accuracy:.2f}%",
        "-" * 40,
        f"Missing/Skipped Tasks:       {missing}",
        "-" * 40
    ]
    
    report = "\n".join(report_lines)
    print(report)
    
    # Save to file
    output_path = Path(args.pred_file).parent / "accuracy_report.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved to: {output_path}")

if __name__ == "__main__":
    main()
