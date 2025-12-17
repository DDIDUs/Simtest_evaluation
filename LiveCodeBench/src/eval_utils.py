import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def index_by_question_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r["question_id"]: r for r in rows if r.get("question_id")}


def find_pred_code(pred_row: Dict[str, Any], target_code_idx: int) -> Optional[Dict[str, Any]]:
    code_results = pred_row.get("code_results", [])
    if not isinstance(code_results, list):
        return None
    for cr in code_results:
        if isinstance(cr, dict) and cr.get("code_index") == target_code_idx:
            return cr
    return None


def get_gt_tc_bools(gt_row: Dict[str, Any], target_code_idx: int) -> Optional[List[Optional[bool]]]:
    gt_eval = gt_row.get("eval_result", [])
    if not isinstance(gt_eval, list) or len(gt_eval) <= target_code_idx:
        return None
    tc_bools = gt_eval[target_code_idx]
    if not isinstance(tc_bools, list):
        return None
    return tc_bools


def decide_num_testcases(
    pred_row: Dict[str, Any],
    gt_tc_bools: List[Optional[bool]],
    pred_pf_list: List[Any],
) -> int:
    n_pred_decl = pred_row.get("num_testcases")
    if isinstance(n_pred_decl, int) and n_pred_decl > 0:
        return n_pred_decl
    if len(gt_tc_bools) > 0:
        return len(gt_tc_bools)
    return len(pred_pf_list)


def pad_or_trim(seq: List[Any], n: int, pad_value: Any = None) -> List[Any]:
    return (seq[:n] + [pad_value] * max(0, n - len(seq)))


def compute_gt_overall(gt_tc_bools: List[Optional[bool]]) -> bool:
    known = [x for x in gt_tc_bools if x is not None]
    return bool(known) and all(x is True for x in known)


def tc_name(i: int, starts_at_0: bool) -> str:
    tc_num = i if starts_at_0 else (i + 1)
    return f"tc_{tc_num}"


def infer_task_level(
    qid: str,
    n: int,
    level_map: Dict[Tuple[str, str], str],
    starts_at_0: bool,
) -> str:
    levels = [level_map.get((qid, tc_name(i, starts_at_0)), "Unknown") for i in range(n)]
    return task_level_from_tc_levels(levels)


def update_level_stats(
    level_stats: Dict[str, Dict[str, int]],
    level: str,
    *,
    task_total: int = 0,
    task_correct: int = 0,
    tc_total: int = 0,
    tc_correct: int = 0,
) -> None:
    st = level_stats.setdefault(
        level, {"total_tasks": 0, "correct_tasks": 0, "total_tcs": 0, "correct_tcs": 0}
    )
    st["total_tasks"] += task_total
    st["correct_tasks"] += task_correct
    st["total_tcs"] += tc_total
    st["correct_tcs"] += tc_correct


def build_report(
    total_tasks: int,
    correct_tasks: int,
    total_tcs: int,
    correct_tcs: int,
    level_stats: Dict[str, Dict[str, int]],
    missing_pred: int,
    missing_gt: int,
) -> str:
    task_acc = (correct_tasks / total_tasks * 100) if total_tasks else 0.0
    tc_acc = (correct_tcs / total_tcs * 100) if total_tcs else 0.0

    lines = [
        "=" * 70,
        "                ACCURACY REPORT (question_id aligned)               ",
        "=" * 70,
        f"{'Metric':<25} | {'Total':<10} | {'Correct':<10} | {'Accuracy':<10}",
        "-" * 70,
        f"{'Overall Task':<25} | {total_tasks:<10} | {correct_tasks:<10} | {task_acc:.2f}%",
        f"{'Overall Test Case':<25} | {total_tcs:<10} | {correct_tcs:<10} | {tc_acc:.2f}%",
        "-" * 70,
    ]

    for level in ["Easy", "Medium", "Hard", "Unknown"]:
        st = level_stats.get(level)
        if not st:
            continue

        lt, lc = st["total_tasks"], st["correct_tasks"]
        ltt, lcc = st["total_tcs"], st["correct_tcs"]
        ltask_acc = (lc / lt * 100) if lt else 0.0
        ltc_acc = (lcc / ltt * 100) if ltt else 0.0

        lines.append(f"{f'[{level}] Task':<25} | {lt:<10} | {lc:<10} | {ltask_acc:.2f}%")
        lines.append(f"{f'[{level}] Test Case':<25} | {ltt:<10} | {lcc:<10} | {ltc_acc:.2f}%")
        lines.append("-" * 70)

    lines.append(f"Missing Pred Rows: {missing_pred}")
    lines.append(f"Missing GT Rows:   {missing_gt}")
    lines.append("=" * 70)
    return "\n".join(lines)


def save_outputs(out_dir: Path, report: str, results: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "accuracy_report.txt").write_text(report + "\n", encoding="utf-8")
    with (out_dir / "accuracy_raw.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def task_level_from_tc_levels(levels):
    order = {"Easy": 1, "Medium": 2, "Hard": 3}
    mx = 0
    best = "Unknown"
    for lv in levels:
        v = order.get(lv, 0)
        if v > mx:
            mx = v
            best = lv
    return best

def load_jsonl_or_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []

    if text[0] == "[":
        obj = json.loads(text)
        if not isinstance(obj, list):
            raise ValueError("Top-level '[' but not a JSON list.")
        return obj

    decoder = json.JSONDecoder()
    idx, n = 0, len(text)
    out = []
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, next_idx = decoder.raw_decode(text, idx)
        out.append(obj)
        idx = next_idx

    for i, r in enumerate(out[:3]):
        if not isinstance(r, dict):
            raise TypeError(f"Parsed item #{i} is not dict: {type(r)}")
    return out

def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def smart_load(path: str):
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    return load_json(path)

def normalize_pred_passfail(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, str):
        s = x.strip().upper()
        if s == "PASS":
            return True
        if s == "FAIL":
            return False
    return False

def build_level_map(level_dir: str):
    level_map = {}
    level_dir = Path(level_dir)

    for level_name in ["easy", "medium", "hard"]:
        fp = level_dir / f"{level_name}.json"
        if not fp.exists():
            continue
        data = load_json(str(fp))
        for s in data.get("ids", []):
            if "#" not in s:
                continue
            qid, tc = s.split("#", 1)
            level_map[(qid, tc)] = level_name.capitalize()
    return level_map
