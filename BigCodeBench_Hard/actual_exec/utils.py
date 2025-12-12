import json
import logging
import os
import re
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
