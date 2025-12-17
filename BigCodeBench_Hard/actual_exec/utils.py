
import json
import ast
import os
import logging
from typing import List, Dict, Any, Optional
from datasets import load_dataset

DEFAULT_DATASET_ID = "bigcode/bigcodebench-hard"
DEFAULT_SPLIT = "v0.1.4"

def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSON or JSONL file and return list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        if not text.strip():
            return []

    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    logging.warning("Skip invalid JSONL line: %s (%s)", line[:80], exc)
        return rows

    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        return list(content.values())
    return []

def load_bigcodebench_hard(dataset_path: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load BigCodeBench-Hard style problems.
     Adapted from actual_exec/utils.py but preserves 'test' field for evaluation.
    """
    records: List[Dict[str, Any]] = []

    if dataset_path and os.path.exists(dataset_path):
        records = read_json_or_jsonl(dataset_path)
    else:
        try:
            ds = load_dataset(dataset_path or DEFAULT_DATASET_ID, split=DEFAULT_SPLIT)
            records = list(ds)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load dataset from {dataset_path or DEFAULT_DATASET_ID} (split={DEFAULT_SPLIT}): {exc}"
            )

    normalized: List[Dict[str, str]] = []
    for idx, item in enumerate(records):
        # Normalize task_id
        task_id = (
            item.get("task_id")
            or item.get("question_id")
            or item.get("problem_id")
            or item.get("id")
            or f"BigCodeBench_{idx}"
        )
        
        # Ensure 'test' field exists - this is critical for evaluation
        test_code = item.get("test")
        if not test_code:
            logging.warning("Skip record without test code: %s", task_id)
            continue

        if task_id == "BigCodeBench/1006":
             # Fix duplicate test case name in BigCodeBench/1006
             # There are two `test_non_zip_content` methods. The first one should be `test_valid_zip_url`.
             test_code = test_code.replace("def test_non_zip_content(self, mock_get):", "def test_valid_zip_url(self, mock_get):", 1)
            
        normalized.append({
            "task_id": str(task_id),
            "test": str(test_code)
        })
            
    return normalized

def split_test_cases(test_case_code):
    """
    Splits a unittest class string into individual test cases.
    Each returned string contains:
    - Global imports and statements
    - The Class definition
    - setUp/tearDown methods (if present)
    - One specific test_ method
    Returns:
    List[Tuple[str, str]]: A list of (test_method_name, full_test_case_code) tuples
    """
    try:
        tree = ast.parse(test_case_code)
    except SyntaxError:
        # Fallback using a dummy name if parsing fails
        return [("error_parsing", test_case_code)]

    # Separating imports/global level stuff from the Class
    imports_and_globals = []
    test_class_node = None
    
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # We assume there is only one main test class usually named TestCases or similar
            # But we'll take the first class we find or specifically look for one inheriting from unittest.TestCase if strictness needed
            # For BigCodeBench, usually it is `class TestCases(unittest.TestCase):`
            test_class_node = node
        else:
            imports_and_globals.append(node)
            

    if not test_class_node:
        return [("no_class_found", test_case_code)]
        
    # Extract setUp, tearDown, and test_ methods
    setup_method = None
    teardown_method = None
    test_methods = []
    other_methods = [] # Helper methods inside the class
    
    for item in test_class_node.body:
        if isinstance(item, ast.FunctionDef):
            if item.name == 'setUp':
                setup_method = item
            elif item.name == 'tearDown':
                teardown_method = item
            elif item.name.startswith('test'):
                test_methods.append(item)
            else:
                other_methods.append(item)
        else:
             # Class docstrings or assignments
            other_methods.append(item)
            

    if not test_methods:
         return [("no_test_methods", test_case_code)]
         
    split_cases = []
    
    # Reconstruct for each test method
    for test_method in test_methods:
        # New class body
        new_body = []
        new_body.extend(other_methods)
        if setup_method:
            new_body.append(setup_method)
        if teardown_method:
            new_body.append(teardown_method)
        new_body.append(test_method)
        
        # Create new ClassDef
        new_class = ast.ClassDef(
            name=test_class_node.name,
            bases=test_class_node.bases,
            keywords=test_class_node.keywords,
            body=new_body,
            decorator_list=test_class_node.decorator_list
        )
        
        # Create new module
        new_module_body = imports_and_globals + [new_class]
        new_module = ast.Module(body=new_module_body, type_ignores=[])
        
        # Unparse to string
        split_cases.append((test_method.name, ast.unparse(new_module)))
        
    return split_cases

if __name__ == "__main__":
    # Path to the file
    RESULTS_PATH = "/home/yrwoo/ICST26/Simtest_evaluation/BigCodeBench_Hard/actual_exec/results/qwen3-coder-30B-A3B-instruct/nucleus_eval_all.json"
    
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: File not found at {RESULTS_PATH}")
        exit(1)
        
    print(f"Loading results from {RESULTS_PATH}...")
    try:
        data = read_json_or_jsonl(RESULTS_PATH)
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)
        
    total_tasks = 0
    passed_tasks = 0
    
    for task in data:
        # Check if code_results exists and matches structure
        # Looking for code_index 0
        code_results = task.get("code_results", [])
        
        # Assuming code_results is a list of results, we need the one with code_index 0
        target_result = None
        for res in code_results:
            if res.get("code_index") == 0:
                target_result = res
                break
        
        # If found, check pass status
        if target_result:
            total_tasks += 1
            # 'overall_pass' seems to be boolean in the file I viewed earlier (e.g. {"code_index": 0, "overall_pass": false})
            # But let's verify if it's string "PASS"/"FAIL" or boolean True/False.
            # In user's `test.jsonl`, it's boolean in `code_results` list: `{"code_index": 0, "overall_pass": false}`
            # BUT in 'overall_pass' top level field it's string "FAIL".
            # The prompt asks for `code_results` check.
            
            is_pass = target_result.get("overall_pass")
            
            # Handle both boolean and string just in case
            if is_pass is True or (isinstance(is_pass, str) and is_pass.upper() == "PASS"):
                passed_tasks += 1
    
if __name__ == "__main__":
    RESULTS_PATH = "/home/yrwoo/ICST26/Simtest_evaluation/BigCodeBench_Hard/actual_exec/results/qwen3-coder-30B-A3B-instruct/nucleus_eval_all.json"
    
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: File not found at {RESULTS_PATH}")
        exit(1)

    # 1. Load Count.json for Ground Truth (if available)
    COUNT_JSON_PATH = "/home/yrwoo/ICST26/Simtest_evaluation/BigCodeBench_Hard/actual_exec/tc_level_index/count.json"
    official_counts = {}
    if os.path.exists(COUNT_JSON_PATH):
        print(f"Loading official counts from {COUNT_JSON_PATH}...")
        try:
            with open(COUNT_JSON_PATH, 'r') as f:
                raw_counts = json.load(f)
                # raw_counts keys are "TaskID#test_method"
                for key in raw_counts:
                    if "#" in key:
                        t_id, tc_name = key.split("#", 1)
                        official_counts[t_id] = official_counts.get(t_id, 0) + 1
        except Exception as e:
            print(f"Error loading count.json: {e}")

    # 2. Load Dataset to get Truth Test Case Counts
    print("Loading dataset to determine test case counts...")
    try:
        dataset_items = load_bigcodebench_hard() 
        task_test_counts = {}
        for item in dataset_items:
            t_id = item['task_id']
            test_code = item['test']
            splits = split_test_cases(test_code)
            
            # Use unique test case names to avoid duplicates (e.g. task 1006)
            unique_tc_names = {name for name, _ in splits if name not in ["error_parsing", "no_class_found", "no_test_methods"]}
            my_count = len(unique_tc_names)
            task_test_counts[t_id] = my_count
            
            # Compare with official if available
            if official_counts:
                off_count = official_counts.get(t_id, 0)
                if my_count != off_count:
                    print(f"DISCREPANCY: Task {t_id} - My Count: {my_count}, Official: {off_count}")
                    # Print the names of split cases to debug
                    print(f"  My Splits: {[name for name, _ in splits]}")
            
        total_counts_check = sum(task_test_counts.values())
        print(f"Loaded {len(task_test_counts)} tasks from dataset. Total test cases (My Count): {total_counts_check}")
        if official_counts:
            print(f"Total test cases (Official Count for these tasks): {sum(official_counts.get(t, 0) for t in task_test_counts)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        task_test_counts = {}

    # 3. Load Results
    print(f"Loading results from {RESULTS_PATH}...")
    try:
        data = read_json_or_jsonl(RESULTS_PATH)
    except Exception as e:
        print(f"Error reading result file: {e}")
        exit(1)
        
    global_total_tests = 0
    global_passed_tests = 0
    
    processed_tasks = 0
    
    for task in data:
        t_id = task.get("question_id") or task.get("task_id")
        
        # Get Code Index 0 Metadata
        if "metadata" not in task or not task["metadata"]:
            continue
            
        # Code index 0
        meta = task["metadata"][0] 
        
        # Determine Total Test Cases for this task
        if t_id in task_test_counts:
            total_tcs = task_test_counts[t_id]
        else:
            # Fallback: Try to use time_breakdown keys + details keys?
            # Or just time_breakdown if available
            tb = meta.get("time_breakdown", {})
            total_tcs = len(tb) if tb else 0
            # If completely failed (no time_breakdown), we map it as 0? 
            # This is risky. Let's warn.
            if total_tcs == 0:
                 # Check if we can parse from graded_list length if it was per-tc? No, graded_list is per-code.
                 pass

        if total_tcs == 0:
             # Skip or warn
             # logging.warning(f"Could not determine total test cases for {t_id}")
             continue
             
        # Determine Failed Test Cases
        # structure: "details": { "test_case_1": "traceback...", ... }
        details = meta.get("details", {})
        
        failed_count = 0
        if isinstance(details, dict):
            # specific test cases failed
            failed_count = len(details)
        else:
            # global failure (e.g. details might be a string or something else?)
            # If status is fail and details is not dict, assume ALL failed?
            # In the user example, details IS a dict even for failure.
            # But just in case:
            if meta.get("status") != "pass":
                 # If detail is string, it's likely global error (syntax error etc) -> All Fail
                 if isinstance(details, str):
                     failed_count = total_tcs
                 elif not details and meta.get("status") == "fail":
                     # Fail status but empty details? Unusual. Assume fail all.
                     failed_count = total_tcs
                 else:
                     # details is dict, counted above
                     pass
        
        # passed = total - failed
        passed_count = total_tcs - failed_count
        if passed_count < 0: passed_count = 0 # Safety
        
        global_total_tests += total_tcs
        global_passed_tests += passed_count
        processed_tasks += 1
        
    if global_total_tests == 0:
        print("No test cases found (or failed to link dataset).")
    else:
        pass_rate = (global_passed_tests / global_total_tests) * 100
        print(f"Processed {processed_tasks} tasks.")
        print(f"Total Test Cases: {global_total_tests}")
        print(f"Passed Test Cases: {global_passed_tests}")
        print(f"Pass Rate: {pass_rate:.2f}%")
