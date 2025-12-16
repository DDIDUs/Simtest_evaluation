import argparse
import json
import logging
import multiprocessing
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Official Evaluation Utils
from bigcodebench_eval_utils import (
    untrusted_check, 
    estimate_pass_at_k, 
    PASS, 
    FAIL, 
    TIMEOUT,
    TIMEOUT_LIMIT
)

from datasets import load_dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "log_code_eval.txt", mode="a"),
        logging.StreamHandler()
    ]
)


STRATEGIES = ["greedy", "nucleus"]
# Use official defaults or overrides
# Official uses 4GB usually for bigcodebench
MAX_AS_LIMIT = 4 * 1024 # MB
MAX_DATA_LIMIT = 8 * 1024 # MB
MAX_STACK_LIMIT = 10 # MB
DEFAULT_DATASET_ID = "bigcode/bigcodebench-hard"
DEFAULT_SPLIT = "v0.1.4"


# --- Data Utilities (Inlined) ---

def ensure_parent_dir(path: str) -> None:
    """Create parent directories for a file path if they do not exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(path: str, data: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def normalize_problem(item: Dict[str, Any], idx: int) -> Optional[Dict[str, str]]:
    """Normalize a dataset row into {task_id, prompt}."""
    task_id = (
        item.get("task_id")
        or item.get("question_id")
        or item.get("problem_id")
        or item.get("id")
        or f"BigCodeBench_{idx}"
    )
    prompt = (
        item.get("complete_prompt")
        or item.get("prompt")
        or item.get("question")
        or item.get("instruction")
        or item.get("text")
        or item.get("problem")
        or item.get("description")
    )
    test = item.get("test")
    entry_point = item.get("entry_point")

    if prompt is None:
        try:
            prompt = json.dumps(item, ensure_ascii=False)
        except Exception:
            prompt = str(item)
    if not prompt:
        return None
        
    return {
        "task_id": str(task_id), 
        "prompt": str(prompt),
        "test": str(test) if test else "",
        "entry_point": str(entry_point) if entry_point else ""
    }


def load_bigcodebench_hard(dataset_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Load BigCodeBench-Hard style problems and normalize fields."""
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
        norm = normalize_problem(item, idx)
        if norm:
            normalized.append(norm)
        else:
            logging.warning("Skip record without prompt: %s", item)
    return normalized


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from an LLM response, preferring fenced blocks."""
    if not text:
        return []

    # ```python ... ```
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if not blocks:
        # Generic fenced block
        blocks = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        # <code>...</code>
        blocks = re.findall(r"<code>(.*?)</code>", text, flags=re.DOTALL | re.IGNORECASE)

    cleaned = [b.strip() for b in blocks if b and b.strip()]
    if cleaned:
        return cleaned

    fallback = text.strip()
    return [fallback] if fallback else []


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
                
                logging.info(f"Processing Task ID: {task_id}")
                
                # Get dynamic test code
                if task_id not in problems_map:
                    logging.warning(f"Task {task_id} not found in dataset. Skipping evaluation for this task.")
                    continue
                
                # Official benchmark uses "test" field which contains the full test harness
                test_code = problems_map[task_id].get("test", "")
                entry_point = problems_map[task_id].get("entry_point", "")
                
                total_tasks += 1
                task_graded_results = []
                task_metadata_list = []
                num_correct = 0

                for code in code_list:
                    # Use official untrusted_check
                    stat, details, runtime, per_test_times = untrusted_check(
                        code=code,
                        test_code=test_code,
                        entry_point=entry_point,
                        max_as_limit=MAX_AS_LIMIT,
                        max_data_limit=MAX_DATA_LIMIT,
                        max_stack_limit=MAX_STACK_LIMIT,
                        gt_time_limit=timeout # Use user provided timeout as GT limit base
                    )
                    
                    is_pass = (stat == PASS)
                    task_graded_results.append(is_pass)
                    
                    # Store metadata similar to before but adapted
                    meta = {
                        "status": stat,
                        "details": details,
                        "runtime": runtime,
                        "time_breakdown": per_test_times
                    }
                    task_metadata_list.append(meta)
                    
                    if is_pass:
                        num_correct += 1
                
                # Greedy Pass@1
                p1_greedy = 1.0 if (task_graded_results and task_graded_results[0]) else 0.0
                pass_at_1_greedy_sum += p1_greedy
                
                # Pass@k Estimator
                est_values = {}
                n_samples = len(code_list)
                for k_val in possible_k:
                    est = estimate_pass_at_k(n_samples, [num_correct], k_val)[0]
                    pass_at_k_sums[k_val] += est
                    est_values[f"pass@{k_val}"] = est

                # Construct detail item
                detail_item = {
                    "question_id": task_id,
                    "code_list": code_list,
                    "graded_list": task_graded_results,
                    "pass@1": p1_greedy,
                    "metadata": task_metadata_list, 
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
    timeout: int = TIMEOUT_LIMIT,
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

    evaluate_files(models_to_use, strategies_to_use, results_root, int(timeout), limit)


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
        type=float,
        default=TIMEOUT_LIMIT,
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
