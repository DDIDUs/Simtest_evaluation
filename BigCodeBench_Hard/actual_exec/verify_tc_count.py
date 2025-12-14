import argparse
import ast
import astunparse
import json
import logging
from typing import List, Dict, Any, Optional
from datasets import load_dataset
# Reuse utils logic directly or import it if path allows.
# Since we are in actual_exec/.., we can try importing from utils if we run from there
# Or just copy the relevant split function to be standalone and safe regardless of path.

from utils import load_bigcodebench_hard, split_test_cases

def main():
    print("Loading BigCodeBench-Hard dataset...")
    ds_data = load_bigcodebench_hard()
    
    # Create Ground Truth Map
    # {task_id: {test_names_set}}
    gt_map = {}
    for item in ds_data:
        t_id = item.get("task_id", "unknown")
        test_code = item.get("test", "")
        split = split_test_cases(test_code)
        valid_tcs = {tc[0] for tc in split if tc[0] not in ("error_parsing", "no_class_found", "no_test_methods")}
        gt_map[t_id] = valid_tcs

    print(f"Loaded {len(gt_map)} items from Dataset.")
    
    # Load Execution Results
    exec_path = "/Users/woo/Documents/Simtest_evaluation/BigCodeBench_Hard/actual_exec/results/qwen3-coder-30B-A3B-instruct/nucleus_eval_all.json"
    print(f"Loading Execution Results from {exec_path}...")
    try:
        with open(exec_path, 'r') as f:
            exec_data = json.load(f)
    except Exception as e:
        print(f"Failed to load exec data: {e}")
        return

    # Create Execution Map
    # {task_id: {test_names_set_from_times}}
    exec_map = {}
    possible_id_fields = ["task_id", "question_id", "id"]
    
    for item in exec_data:
        t_id = None
        for field in possible_id_fields:
            if field in item:
                t_id = item[field]
                break
        
        if not t_id:
            continue
            
        metadata_list = item.get("metadata", [])
        executed_tests = set()
        
        # Aggregate from all samples just in case
        for meta in metadata_list:
            if not meta: continue
            times_map = meta.get("times", {})
            for key in times_map.keys():
                # key might be "candidate.TestCases.test_case_1"
                simple_key = key.split('.')[-1]
                executed_tests.add(simple_key)
        
        exec_map[t_id] = executed_tests

    # Compare
    total_gt = 0
    total_exec = 0
    
    print("\n--- Discrepancy Report ---")
    mismatch_count = 0
    
    for t_id, gt_tests in gt_map.items():
        exec_tests = exec_map.get(t_id, set())
        
        total_gt += len(gt_tests)
        total_exec += len(exec_tests)
        
        if len(gt_tests) != len(exec_tests):
            mismatch_count += 1
            print(f"ID: {t_id}")
            print(f"  GT Count: {len(gt_tests)} | Exec Count: {len(exec_tests)}")
            
            missing_in_exec = gt_tests - exec_tests
            extra_in_exec = exec_tests - gt_tests
            
            if missing_in_exec:
                print(f"  Missing in Exec: {list(missing_in_exec)[:5]}...")
            if extra_in_exec:
                print(f"  Extra in Exec:   {list(extra_in_exec)[:5]}...")
            print("-" * 20)

    print("\n--- Summary ---")
    print(f"Total GT Test Cases:   {total_gt}")
    print(f"Total Exec Test Cases: {total_exec}")
    print(f"Tasks with Mismatch:   {mismatch_count}")


if __name__ == "__main__":
    main()
