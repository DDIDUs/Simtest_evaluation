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
        
        # specific prediction
        pred_status = pred.get('overall_pass')
        pred_bool = (pred_status == "PASS")
        
        # ground truth
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
            
        results.append({
            "id": t_id,
            "code_index": code_idx,
            "prediction": pred_status,
            "ground_truth": "PASS" if actual_bool else "FAIL",
            "is_correct": is_correct
        })
        
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    
    report_lines = [
        "-" * 40,
        f"Total Predictions Evaluated: {total}",
        f"Correct Predictions: {correct}",
        f"Missing/Skipped: {missing}",
        f"Accuracy: {accuracy:.2f}%",
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
