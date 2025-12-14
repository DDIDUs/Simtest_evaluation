import json
import logging
import os
import re
import ast
import astunparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset


DEFAULT_DATASET_ID = "bigcode/bigcodebench-hard"
DEFAULT_SPLIT = "v0.1.4"


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


def extract_first_code(text: str) -> Optional[str]:
    blocks = extract_code_blocks(text)
    return blocks[0] if blocks else None

def split_test_cases(test_case_code: str) -> List[tuple]:
    """
    Splits a unittest class string into individual test cases.
    Returns: List[Tuple[str, str]] (test_method_name, full_test_case_code)
    """
    try:
        tree = ast.parse(test_case_code)
    except SyntaxError:
        return [("error_parsing", test_case_code)]

    # Separating imports/global level stuff from the Class
    imports_and_globals = []
    test_class_node = None
    
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            test_class_node = node
        else:
            imports_and_globals.append(node)
            
    if not test_class_node:
        return [("no_class_found", test_case_code)]
        
    # Extract setUp, tearDown, and test_ methods
    setup_method = None
    teardown_method = None
    test_methods = []
    other_methods = [] 
    
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
            other_methods.append(item)
            
    if not test_methods:
         return [("no_test_methods", test_case_code)]
         
    split_cases = []
    
    for test_method in test_methods:
        new_body = []
        new_body.extend(other_methods)
        if setup_method:
            new_body.append(setup_method)
        if teardown_method:
            new_body.append(teardown_method)
        new_body.append(test_method)
        
        new_class = ast.ClassDef(
            name=test_class_node.name,
            bases=test_class_node.bases,
            keywords=test_class_node.keywords,
            body=new_body,
            decorator_list=test_class_node.decorator_list
        )
        
        new_module_body = imports_and_globals + [new_class]
        new_module = ast.Module(body=new_module_body, type_ignores=[])
        
        split_cases.append((test_method.name, astunparse.unparse(new_module)))
        
    return split_cases
