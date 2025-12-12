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

from utils import extract_code_blocks, save_json

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
    filename=Path(__file__).parent / "log_code_eval.txt",
    filemode="a",
)


STRATEGIES = ["greedy", "neuclus"]
GLOBAL_TIMEOUT = 30  # seconds per code


# --- Test runner logic (adapted from compute_code_generation_metrics style) ---
def _run_suite(code_str: str, result_list, metadata_list) -> None:
    try:
        run_result, run_times, meta = run_tests_for_code(code_str)
    except Exception as exc:  # safeguard against unexpected runner failures
        run_result = [False]
        run_times = []
        meta = {"error": repr(exc), "trace": ""}
    result_list.append((run_result, run_times))
    metadata_list.append(meta)


def check_correctness(code_str: str, timeout: int) -> Tuple[List[bool], Dict]:
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(target=_run_suite, args=(code_str, result, metadata_list))
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


def run_tests_for_code(code_str: str) -> Tuple[List[bool], List[float], Dict]:
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
                exec(code_str, module.__dict__)
            task_func = module.__dict__.get("task_func")
            if not callable(task_func):
                raise ValueError("task_func is not defined in generated code.")

            class TestCases(unittest.TestCase):
                def setUp(self):
                    if not os.path.exists("downloaded_files"):
                        os.makedirs("downloaded_files")

                def tearDown(self):
                    if os.path.exists("downloaded_files"):
                        for filename in os.listdir("downloaded_files"):
                            os.remove(os.path.join("downloaded_files", filename))
                        os.rmdir("downloaded_files")

                @patch("ftplib.FTP")
                @patch("subprocess.call")
                def test_case_1(self, mock_subprocess_call, mock_ftp):
                    mock_ftp.return_value.nlst.return_value = ["file1.txt", "file2.jpg"]
                    mock_subprocess_call.return_value = 0
                    downloaded_files = task_func()
                    self.assertEqual(len(downloaded_files), 2)
                    self.assertIn("file1.txt", downloaded_files)
                    self.assertIn("file2.jpg", downloaded_files)

                @patch("ftplib.FTP")
                def test_case_2(self, mock_ftp):
                    error_message = "Failed to connect to FTP server"
                    mock_ftp.side_effect = Exception(error_message)
                    with self.assertRaises(Exception) as context:
                        task_func(ftp_server="invalid_server")
                    self.assertEqual(
                        str(context.exception),
                        f"Failed to connect to FTP server invalid_server: {error_message}",
                    )

                @patch("ftplib.FTP")
                def test_case_3(self, mock_ftp):
                    error_message = "Failed to login"
                    mock_ftp.return_value.login.side_effect = Exception(error_message)
                    with self.assertRaises(Exception) as context:
                        task_func(ftp_user="invalid_user")
                    self.assertEqual(
                        str(context.exception),
                        f"Failed to log into FTP server ftp.dlptest.com with user invalid_user: {error_message}",
                    )

                @patch("ftplib.FTP")
                def test_case_4(self, mock_ftp):
                    error_message = "Failed to login"
                    mock_ftp.return_value.login.side_effect = Exception(error_message)
                    with self.assertRaises(Exception) as context:
                        task_func(ftp_password="invalid_password")
                    self.assertEqual(
                        str(context.exception),
                        f"Failed to log into FTP server ftp.dlptest.com with user dlpuser: {error_message}",
                    )

                @patch("ftplib.FTP")
                def test_case_5(self, mock_ftp):
                    error_message = "Failed to change directory"
                    mock_ftp.return_value.cwd.side_effect = Exception(error_message)
                    with self.assertRaises(Exception) as context:
                        task_func(ftp_dir="/invalid_directory")
                    self.assertEqual(
                        str(context.exception),
                        f"Failed to change to directory /invalid_directory on server ftp.dlptest.com: {error_message}",
                    )

            class TimeTrackingTestResult(unittest.TextTestResult):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.test_times = []
                    self._current_start_time = 0

                def startTest(self, test):
                    self._current_start_time = time.time()
                    super().startTest(test)

                def stopTest(self, test):
                    elapsed = time.time() - self._current_start_time
                    self.test_times.append(elapsed)
                    super().stopTest(test)

            suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCases)
            runner = unittest.TextTestRunner(
                stream=buf, verbosity=2, resultclass=TimeTrackingTestResult
            )
            result = runner.run(suite)
            # For precise per-test booleans, reconstruct using result data
            total_tests = suite.countTestCases()
            per_test_flags = [True] * total_tests
            # Note: result.test_times populated by TimeTrackingTestResult
            per_test_times = getattr(result, "test_times", [0.0] * total_tests)

            for failed, _ in result.failures:
                idx = int(failed.id().split(".")[-1].replace("test_case_", "")) - 1
                per_test_flags[idx] = False
            for err, _ in result.errors:
                idx = int(err.id().split(".")[-1].replace("test_case_", "")) - 1
                per_test_flags[idx] = False

            # Parse failures/errors to map to specific test cases
            error_map = {}
            for failure in result.failures:
                # Expected format: "test_case_X (TestCases.test_case_X)" or similar depending on unittest
                # We extracted test_case_N earlier manually
                # result.failures is list of (test_case, msg)
                case_id = failure[0].id().split(".")[-1]
                msg = failure[1]
                error_map[case_id] = msg
            
            for error in result.errors:
                case_id = error[0].id().split(".")[-1]
                msg = error[1]
                error_map[case_id] = msg

            meta = {
                "failures": error_map,
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
    for model_name in models:
        for strategy in strategies:
            gen_path = Path(results_root) / model_name / f"{strategy}_code_generate.json"
            if not gen_path.exists():
                logging.warning("Skip missing generation file: %s", gen_path)
                continue

            codes = load_generated_codes(str(gen_path))
            # Prepare data structures for detail and summary
            
            # eval_all.json list
            detail_results = []
            
            # For summary stats
            total_tasks = 0
            pass_at_1_greedy_sum = 0.0
            
            # We will calculate pass@k for these k values if n_samples >= k
            # Determine max_samples from data
            max_samples = 0
            if codes:
                max_samples = max(len(cl) for cl in codes.values())
            
            # Standard k values to check + the actual max samples if distinct
            target_k_values = {1, 5, 10}
            if max_samples > 0:
                target_k_values.add(max_samples)
            
            possible_k = sorted([k for k in target_k_values if k <= max_samples])
            pass_at_k_sums = {k: 0.0 for k in possible_k}

            for task_id, code_list in list(codes.items())[: limit or None]:
                total_tasks += 1
                graded_list = []
                num_correct = 0
                
                # Check all samples
                for idx, code in enumerate(code_list):
                    try:
                        (res, times), meta = check_correctness(code, timeout=timeout)
                        is_pass = all(res)
                        graded_list.append(is_pass)
                        if is_pass:
                            num_correct += 1
                    except Exception as exc:
                        # Should be handled inside check_correctness but catch-all here
                        graded_list.append(False)

                # Greedy pass@1: Correctness of the FIRST sample (idx 0)
                # If no code generated, count as 0
                p1_greedy = 1.0 if (graded_list and graded_list[0]) else 0.0
                pass_at_1_greedy_sum += p1_greedy
                
                # Unbiased estimator for pass@k
                n_samples = len(code_list)
                task_pass_at_k = {}
                for k_val in possible_k:
                    est = estimate_pass_at_k(n_samples, num_correct, k_val)
                    task_pass_at_k[f"pass@{k_val}"] = est
                    pass_at_k_sums[k_val] += est

                # Detail entry
                # We need to re-run or store metadata for detailed error report? 
                # Ideally we stored it above. Optimization: check_correctness is expensive so we run it once per code.
                # To match the requested format, we want "failures" and "errors" details.
                # Re-running check_correctness just for metadata if it failed?
                # Actually, in the loop above we didn't store metadata. Let's fix loop to store it.
                
                # Refactoring loop to store results
                pass # Placeholder to allow re-writing the block below correctly
                
            # Re-implementing the loop properly
            detail_results = []
            total_tasks = 0
            pass_at_1_greedy_sum = 0.0
            pass_at_k_sums = {k: 0.0 for k in possible_k}
            
            for task_id, code_list in list(codes.items())[: limit or None]:
                total_tasks += 1
                task_graded_results = []
                task_metadata_list = []
                num_correct = 0

                for code in code_list:
                    (res, times), meta = check_correctness(code, timeout=timeout)
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

            # Calculate Summary
            summary = {
                "total": total_tasks,
                "pass@1_greedy": pass_at_1_greedy_sum / total_tasks if total_tasks else 0.0
            }
            for k_val in possible_k:
                summary[f"pass@{k_val}"] = pass_at_k_sums[k_val] / total_tasks if total_tasks else 0.0

            # Save Eval (Summary)
            out_path_summary = Path(results_root) / model_name / f"{strategy}_eval.json"
            save_json(str(out_path_summary), summary)
            
            # Save Eval All (Details)
            out_path_details = Path(results_root) / model_name / f"{strategy}_eval_all.json"
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
        help="Specific strategies to evaluate (greedy, neuclus). Default: all.",
    )
    args = parser.parse_args()

    main(args.results_root, args.models, args.timeout, args.limit, args.sampling)
