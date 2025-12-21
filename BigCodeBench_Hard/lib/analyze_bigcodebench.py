import json
import os
from collections import Counter, defaultdict

# Directory containing results
# Script is in BigCodeBench_Hard/lib
# Results are in BigCodeBench_Hard/1_no_reasoning/results
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BigCodeBench_Hard/lib/../1_no_reasoning/results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "1_no_reasoning", "results")

def analyze_model(model_name):
    file_path = os.path.join(RESULTS_DIR, model_name, "accuracy_raw.jsonl")
    if not os.path.exists(file_path):
        print(f"Skipping {model_name}: File not found")
        return

    all_pass_count = 0
    all_fail_count = 0
    all_pass_categories = Counter()
    all_fail_categories = Counter()
    
    try:
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                task_id = data.get("task_id")
                code_category = data.get("code_category", "Unknown")
                test_cases = data.get("test_cases", {})
                
                if not test_cases:
                    continue

                preds = [tc.get("pred") for tc in test_cases.values()]
                
                # Check for All Pass
                if all(p == "PASS" for p in preds):
                    all_pass_count += 1
                    all_pass_categories[code_category] += 1
                
                # Check for All Fail
                if all(p == "FAIL" for p in preds):
                    all_fail_count += 1
                    all_fail_categories[code_category] += 1

        print(f"### Model: {model_name}")
        print(f"Total Consistent Cases (Total All-Pass + Total All-Fail): {all_pass_count + all_fail_count}")
        
        print(f"**All Test Cases FAIL**: {all_fail_count} tasks")
        if all_fail_count > 0:
            print("  Category Breakdown:")
            for cat, count in all_fail_categories.items():
                print(f"    - {cat}: {count}")
        
        print(f"**All Test Cases PASS**: {all_pass_count} tasks")
        if all_pass_count > 0:
            print("  Category Breakdown:")
            for cat, count in all_pass_categories.items():
                print(f"    - {cat}: {count}")
        print("\n" + "="*40 + "\n")

    except Exception as e:
        print(f"Error processing {model_name}: {e}")

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Directory {RESULTS_DIR} not found.")
        return

    models = sorted(os.listdir(RESULTS_DIR))
    for model in models:
        # Skip if not a directory
        if not os.path.isdir(os.path.join(RESULTS_DIR, model)):
            continue
        analyze_model(model)

if __name__ == "__main__":
    main()
