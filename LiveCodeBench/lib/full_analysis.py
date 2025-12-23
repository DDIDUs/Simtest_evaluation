import os
import json
from collections import defaultdict

# Base directory
BASE_DIR = "/home/yrwoo/ICST26/Simtest_evaluation/LiveCodeBench"

# Strategies
STRATEGIES = [
    "1_pred",
    "2_bug_local", 
    "3_bug_report"
]

def get_available_models(base_dir, strategies):
    """Discover models by checking the first strategy's results directory."""
    # LCB Structure: results/{strategy}/{model}/...
    first_strategy = strategies[0]
    results_dir = os.path.join(base_dir, "results", first_strategy)
    
    if not os.path.exists(results_dir):
        return []
        
    models = []
    for name in os.listdir(results_dir):
        if os.path.isdir(os.path.join(results_dir, name)):
            models.append(name)
    return sorted(models)

def calculate_full_metrics(base_dir, strategies):
    models = get_available_models(base_dir, strategies)
    
    if not models:
        print("No models found!")
        return

    output_file = os.path.join(base_dir, "lib", "full_analysis_report.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Full Dataset Confusion Analysis\n")
        f.write(f"Benchmark: LiveCodeBench\n")
        f.write(f"{'='*60}\n\n")

        for model in models:
            f.write(f"{'#'*60}\n")
            f.write(f"Model: {model}\n")
            f.write(f"{'#'*60}\n")
            
            for strategy in strategies:
                # LCB Path: results/{strategy}/{model}/accuracy_raw.jsonl
                file_path = os.path.join(base_dir, "results", strategy, model, "accuracy_raw.jsonl")
                
                if not os.path.exists(file_path):
                    f.write(f"\nScanning Strategy: {strategy}\n")
                    f.write(f"  [!] File not found: {file_path}\n")
                    continue

                # Process File
                tp = 0
                fp = 0
                tn = 0
                fn = 0
                
                try:
                    with open(file_path, 'r') as json_f:
                        for line in json_f:
                            if not line.strip():
                                continue
                            task = json.loads(line)
                            test_cases = task.get("test_cases", {})
                            
                            for tc_key, tc_data in test_cases.items():
                                status = tc_data.get("gt", "FAIL") # Actual (gt for LCB)
                                pred = tc_data.get("pred", "FAIL")   # Prediction
                                
                                if pred == "PASS":
                                    if status == "PASS":
                                        tp += 1
                                    else:
                                        fp += 1
                                else: # pred == "FAIL"
                                    if status == "FAIL":
                                        tn += 1
                                    else: # status == "PASS"
                                        fn += 1
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    
                    f.write(f"\nScanning Strategy: {strategy}\n")
                    f.write(f"Precision: {precision:.4f}\n")
                    f.write(f"               Prediction\n")
                    f.write(f"             PASS  |  FAIL\n")
                    f.write(f"----------------------------\n")
                    f.write(f"Actual PASS | {tp:<5} | {fn:<5}\n")
                    f.write(f"       FAIL | {fp:<5} | {tn:<5}\n")
                    
                except Exception as e:
                    f.write(f"\nScanning Strategy: {strategy}\n")
                    f.write(f"  [!] Error processing file: {e}\n")

    print(f"Full analysis report saved to: {output_file}")

if __name__ == "__main__":
    calculate_full_metrics(BASE_DIR, STRATEGIES)
