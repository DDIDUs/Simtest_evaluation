import argparse
import io
import json
import logging
import multiprocessing
import os
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

from utils import extract_code_blocks, save_json, load_bigcodebench_hard

import numpy as np

def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """
    Estimates pass@k using the unbiased estimator.
    Ref: https://arxiv.org/abs/2107.03374
    """
    if num_samples < k:
        return 0.0
    if num_correct == num_samples:
        return 1.0

    # 1 - binom(n-c, k) / binom(n, k)
    # Calculated as: 1 - prod_{i=0}^{k-1} ( (n - c - i) / (n - i) )
    
    n = num_samples
    c = num_correct
    
    prob_failure = 1.0
    for i in range(k):
        prob_failure *= (n - c - i) / (n - i)
    
    return 1.0 - prob_failure



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "log_code_eval.txt", mode="a"),
        logging.StreamHandler()
    ]
)


STRATEGIES = ["greedy", "nucleus"]
GLOBAL_TIMEOUT = 30  # seconds per code


# --- Test runner logic (adapted from compute_code_generation_metrics style) ---
def _run_suite(code_str: str, test_code: str, result_list, metadata_list) -> None:
    try:
        run_result, run_times, meta = run_tests_for_code(code_str, test_code)
    except Exception as exc:  # safeguard against unexpected runner failures
        run_result = [False]
        run_times = []
        meta = {"error": repr(exc), "trace": ""}
    result_list.append((run_result, run_times))
    metadata_list.append(meta)


def check_correctness(code_str: str, test_code: str, timeout: int) -> Tuple[List[bool], Dict]:
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(target=_run_suite, args=(code_str, test_code, result, metadata_list))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.kill()

    if not result:
        # Timeout or crash
        result.append(([False], []))
        metadata_list.append(
            {"error": "GlobalTimeout", "trace": "Global timeout exceeded in check_correctness"}
        )

    return result[0], metadata_list[0]


def run_tests_for_code(code_str: str, test_code: str) -> Tuple[List[bool], Dict, Dict]:
    """Execute provided unit tests against a single code string and return per-test pass list."""
    buf = io.StringIO()

    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            module = types.ModuleType("candidate")
            # If the code requests input, we raise SystemExit to kill this test process
            # (check_correctness handles process death gracefully)
            with patch("builtins.input", side_effect=SystemExit("Input called")), \
                 patch("getpass.getpass", side_effect=SystemExit("Getpass called")):
                
                # Exec candidate code
                exec(code_str, module.__dict__)
                
                # Exec test code in the same module dict so it can see the candidate functions
                exec(test_code, module.__dict__)
            
            # Find the test class
            TestClass = None
            for name, val in module.__dict__.items():
                if isinstance(val, type) and issubclass(val, unittest.TestCase):
                    TestClass = val
                    break
            
            if not TestClass:
                 # Fallback: maybe specific class name expected like "TestCases"?
                 TestClass = module.__dict__.get("TestCases")
            
            if not TestClass:
                raise ValueError("No unittest.TestCase class found in the provided test code.")

            class TimeTrackingTestResult(unittest.TextTestResult):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.test_times = {}
                    self._current_start_time = 0

                def startTest(self, test):
                    self._current_start_time = time.time()
                    super().startTest(test)

                def stopTest(self, test):
                    elapsed = time.time() - self._current_start_time
                    self.test_times[test.id()] = elapsed
                    super().stopTest(test)

            suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestClass)
            runner = unittest.TextTestRunner(
                stream=buf, verbosity=2, resultclass=TimeTrackingTestResult
            )
            # Match test cases to indices by iterating over the suite
            test_id_to_index = {}
            for i, test in enumerate(suite):
                test_id_to_index[test.id()] = i

            result = runner.run(suite)
            
            # Initialize flags
            total_tests = suite.countTestCases()
            per_test_flags = [True] * total_tests
            
            # Get times (now a dict)
            per_test_times = getattr(result, "test_times", {})

            # Helper to mark failures
            def mark_failure(test_obj, reason):
                # Try exact ID match first
                tid = test_obj.id()
                if tid in test_id_to_index:
                    per_test_flags[test_id_to_index[tid]] = False
                    return
                # Fallback: try parsing test_case_N
                try:
                    # Expecting format like "test_case_1 (TestCases.test_case_1)"
                    # test_obj.id() gives "module.Class.method"
                    method_name = tid.split(".")[-1]
                    if method_name.startswith("test_case_"):
                        idx = int(method_name.replace("test_case_", ""))
                        if 0 <= idx < total_tests:
                            per_test_flags[idx] = False
                except ValueError:
                    pass

            for failed, _ in result.failures:
                mark_failure(failed, "failure")
            for err, _ in result.errors:
                mark_failure(err, "error")

            # Final safeguard: if result says fail but we didn't mark any flag (mismatch), mark all as False
            if not result.wasSuccessful() and all(per_test_flags):
                per_test_flags = [False] * total_tests

            # Parse failures/errors map
            error_map = {}
            for failure in result.failures:
                case_id = failure[0].id().split(".")[-1]
                msg = failure[1]
                error_map[case_id] = msg
            
            for error in result.errors:
                case_id = error[0].id().split(".")[-1]
                msg = error[1]
                error_map[case_id] = msg

            meta = {
                "failures": error_map,
                "times": per_test_times,
                "trace": buf.getvalue(),
            }
            return per_test_flags, per_test_times, meta
        finally:
            os.chdir(cwd)


# --- Loading and evaluation orchestration ---
def load_generated_codes(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    codes: Dict[str, List[str]] = {}
    for task_id, parsed_list in (data.get("code") or {}).items():
        if parsed_list:
            codes[task_id] = parsed_list
    for task_id, raw_list in (data.get("raw") or {}).items():
        if raw_list:
            # Only extract from raw if we don't already have codes for this task
            if task_id not in codes:
                extracted: List[str] = []
                for raw in raw_list:
                    extracted.extend(extract_code_blocks(raw))
                codes[task_id] = extracted
    return codes


def evaluate_files(
    models: List[str],
    strategies: List[str],
    results_root: str,
    timeout: int,
    limit: Optional[int] = None,
) -> None:
    # Load dataset once to get test codes
    logging.info("Loading BigCodeBench-Hard dataset...")
    problems_list = load_bigcodebench_hard()
    problems_map = {p["task_id"]: p for p in problems_list}

    for model_name in models:
        for strategy in strategies:
            gen_filename = f"{strategy}_code_generate.json"
            gen_path = Path(results_root) / model_name / gen_filename
            
            if strategy == "nucleus" and not gen_path.exists():
                 fallback_path = Path(results_root) / model_name / "nuclus_code_generate.json"
                 if fallback_path.exists():
                     logging.info(f"Using fallback file: {fallback_path}")
                     gen_path = fallback_path

            if not gen_path.exists():
                logging.warning("Skip missing generation file: %s", gen_path)
                continue

            codes = load_generated_codes(str(gen_path))
            # Prepare data structures
            detail_results = []
            processed_ids = set()
            
            # Paths
            out_path_summary = Path(results_root) / model_name / f"{strategy}_eval.json"
            out_path_details = Path(results_root) / model_name / f"{strategy}_eval_all.json"

            # Resume logic: Load existing details if available
            if out_path_details.exists():
                try:
                    with open(out_path_details, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, list):
                            detail_results = existing_data
                            processed_ids = {item["question_id"] for item in detail_results}
                            logging.info(f"Resuming evaluation. Found {len(detail_results)} existing records.")
                except Exception as e:
                    logging.warning(f"Could not load existing evaluation details for resuming: {e}")

            # Calculate metrics counters based on existing data
            total_tasks = len(detail_results)
            pass_at_1_greedy_sum = sum(item["pass@1"] for item in detail_results)
            
            # Determine max_samples from data
            max_samples = 0
            if codes:
                max_samples = max(len(cl) for cl in codes.values())
            
            # Standard k values
            target_k_values = {1, 5, 10}
            if max_samples > 0:
                target_k_values.add(max_samples)
            
            possible_k = sorted([k for k in target_k_values if k <= max_samples])
            
            # Reconstruct pass_at_k sums from existing details
            pass_at_k_sums = {k: 0.0 for k in possible_k}
            for item in detail_results:
                for k_val in possible_k:
                    k_key = f"pass@{k_val}"
                    if k_key in item:
                        pass_at_k_sums[k_val] += item[k_key]

            code_items = list(codes.items())[: limit or None]
            total_count = len(code_items)
            logging.info(f"Starting evaluation for {model_name} - {strategy}: {total_count} tasks total. (Already done: {len(processed_ids)})")

            processed_this_run = 0
            save_interval = 1

            for task_id, code_list in code_items:
                # Skip if already processed
                if task_id in processed_ids:
                    continue

                processed_this_run += 1
                if processed_this_run % 1 == 0:
                    logging.info(f"Evaluated {processed_this_run} new tasks (Total: {len(detail_results) + 1}/{total_count})...")
                
                # Get dynamic test code
                if task_id not in problems_map:
                    logging.warning(f"Task {task_id} not found in dataset. Skipping evaluation for this task.")
                    continue
                test_code = problems_map[task_id].get("test", "")
                
                total_tasks += 1
                task_graded_results = []
                task_metadata_list = []
                num_correct = 0

                for code in code_list:
                    (res, times), meta = check_correctness(code, test_code, timeout=timeout)
                    is_pass = all(res)
                    task_graded_results.append(is_pass)
                    task_metadata_list.append(meta) # Store full metadata including errors
                    if is_pass:
                        num_correct += 1
                
                # Greedy Pass@1
                p1_greedy = 1.0 if (task_graded_results and task_graded_results[0]) else 0.0
                pass_at_1_greedy_sum += p1_greedy
                
                # Pass@k Estimator
                est_values = {}
                n_samples = len(code_list)
                for k_val in possible_k:
                    est = estimate_pass_at_k(n_samples, num_correct, k_val)
                    pass_at_k_sums[k_val] += est
                    est_values[f"pass@{k_val}"] = est

                # Construct detail item
                detail_item = {
                    "question_id": task_id, # Mapping task_id to question_id
                    "code_list": code_list,
                    "graded_list": task_graded_results,
                    "pass@1": p1_greedy,
                    "metadata": task_metadata_list, # List of dicts
                    **est_values 
                }
                detail_results.append(detail_item)
                
                # Incremental Save
                if processed_this_run % save_interval == 0:
                    # Summary
                    summary = {
                        "total": total_tasks,
                        "pass@1_greedy": pass_at_1_greedy_sum / total_tasks if total_tasks else 0.0
                    }
                    for k_val in possible_k:
                        summary[f"pass@{k_val}"] = pass_at_k_sums[k_val] / total_tasks if total_tasks else 0.0
                    
                    save_json(str(out_path_summary), summary)
                    save_json(str(out_path_details), detail_results)
                    logging.info(f"Saved incremental progress: {len(detail_results)}/{total_count}")

            # Final Save
            summary = {
                "total": total_tasks,
                "pass@1_greedy": pass_at_1_greedy_sum / total_tasks if total_tasks else 0.0
            }
            for k_val in possible_k:
                summary[f"pass@{k_val}"] = pass_at_k_sums[k_val] / total_tasks if total_tasks else 0.0

            save_json(str(out_path_summary), summary)
            save_json(str(out_path_details), detail_results)
            
            logging.info("Saved eval summary: %s", out_path_summary)
            logging.info("Saved eval details: %s", out_path_details)


def main(
    results_root: str,
    models: Optional[List[str]] = None,
    timeout: int = GLOBAL_TIMEOUT,
    limit: Optional[int] = None,
    sampling: Optional[List[str]] = None,
) -> None:
    models_to_use = models or [
        "gpt-4o-2024-08-06",
        "qwenn3-coder-30B-A3B-instruct",
    ]
    strategies_to_use = STRATEGIES
    if sampling:
        strategies_to_use = [s for s in STRATEGIES if s in sampling]

    evaluate_files(models_to_use, strategies_to_use, results_root, timeout, limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        type=str,
        default=str(Path(__file__).parent / "results"),
        help="Root directory where generation outputs are stored.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Specific models to evaluate (default: both).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=GLOBAL_TIMEOUT,
        help="Per-code global timeout (seconds).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of tasks to evaluate (e.g., 3 for dry run).",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        nargs="*",
        default=None,
        help="Specific strategies to evaluate (greedy, nucleus). Default: all.",
    )
    args = parser.parse_args()

    main(args.results_root, args.models, args.timeout, args.limit, args.sampling)
