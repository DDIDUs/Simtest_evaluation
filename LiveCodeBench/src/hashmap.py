#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


def to_bin_10(x: float) -> int:
    if x <= 0.0:
        return 0
    if x >= 1.0:
        return 9
    b = int(math.floor(x * 10.0))
    return min(max(b, 0), 9)


def bin10_to_level(b: int) -> int:
    """
    10-bin index -> 3-level index
    0: Hard   (0.0 ~ 0.1)
    1: Medium (0.1 ~ 0.9)
    2: Easy   (0.9 ~ 1.0)
    """
    if b <= 0:
        return 0
    if b >= 9:
        return 2
    return 1


def mean_bool(arr: List[bool]) -> float:
    if not arr:
        return 0.0
    return sum(1 for v in arr if v) / float(len(arr))


def extract_qid_from_task_id(task_id: str) -> Optional[str]:
    nums = re.findall(r"\d+", task_id or "")
    if not nums:
        return None
    return nums[-1]


def load_samples_json(samples_path: str) -> List[Dict[str, Any]]:
    with open(samples_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str):
    """
    indent(멀티라인) JSON 객체를 순차적으로 yield.
    기존 코드 호환을 위해 (end_line_no, obj) 형태로 yield.
    """
    buf = []
    depth = 0
    in_obj = False
    start_line_no = None

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            if not in_obj:
                if line.lstrip().startswith("{"):
                    in_obj = True
                    start_line_no = line_no
                    buf = []
                    depth = 0
                else:
                    continue

            buf.append(line)
            depth += line.count("{")
            depth -= line.count("}")

            if in_obj and depth == 0:
                raw = "".join(buf)
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"JSON parse error (lines {start_line_no}-{line_no}): {e}"
                    ) from e

                yield line_no, obj
                in_obj = False
                buf = []
                start_line_no = None


def build_axes_map(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    axes_map: Dict[str, Dict[str, Any]] = {}

    for s in samples:
        qid = str(s.get("question_id"))
        eval_result = s.get("eval_result")

        if not isinstance(eval_result, list) or len(eval_result) == 0:
            continue
        if not isinstance(eval_result[0], list) or len(eval_result[0]) == 0:
            continue

        n_codes = len(eval_result)
        for row in eval_result:
            if not isinstance(row, list):
                raise ValueError(f"Invalid eval_result row type for qid={qid}")

        n_tcs = min(len(row) for row in eval_result)

        # X: tc별 code 통과 비율
        x_vals: List[float] = []
        for j in range(n_tcs):
            pass_cnt = 0
            for i in range(n_codes):
                v = eval_result[i][j]
                if isinstance(v, bool) and v:
                    pass_cnt += 1
                elif v in (1, "true", "True"):
                    pass_cnt += 1
            x_vals.append(pass_cnt / float(n_codes) if n_codes else 0.0)

        x_bins = [to_bin_10(x) for x in x_vals]

        # Y: code0 overall pass ratio
        code0 = eval_result[0][:n_tcs]
        y_val = mean_bool([bool(v) for v in code0])
        y_bin = to_bin_10(y_val)

        axes_map[qid] = {
            "x": x_vals,
            "x_bins": x_bins,
            "y": y_val,
            "y_bin": y_bin,
            "n_codes": n_codes,
            "n_tcs": n_tcs,
        }

    return axes_map


def sort_test_case_keys(test_cases: Dict[str, Any]) -> List[str]:
    def key_fn(k: str) -> int:
        m = re.search(r"(\d+)$", k)
        return int(m.group(1)) if m else 10**9
    return sorted(test_cases.keys(), key=key_fn)


def aggregate_correct_by_bins(
    axes_map: Dict[str, Dict[str, Any]],
    jsonl_path: str,
):
    """
    3x3 grid로 집계:
      cell = (y_level, x_level) where each in {0,1,2}
    """
    global_grid = defaultdict(lambda: {"total": 0, "correct_true": 0, "correct_false": 0})
    per_question = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct_true": 0, "correct_false": 0}))

    for _, obj in iter_jsonl(jsonl_path):
        qid = obj.get("question_id")
        if qid is None:
            qid = extract_qid_from_task_id(str(obj.get("task_id", "")))
        if qid is None:
            continue
        qid = str(qid)

        axes = axes_map.get(qid)
        if axes is None:
            continue

        test_cases = obj.get("test_cases", {})
        if not isinstance(test_cases, dict) or not test_cases:
            continue

        ordered_keys = sort_test_case_keys(test_cases)
        n = min(int(axes["n_tcs"]), len(ordered_keys))

        # 10-bin -> 3-level
        y_level = bin10_to_level(int(axes["y_bin"]))
        x_bins = axes["x_bins"]

        for j in range(n):
            tc = test_cases.get(ordered_keys[j], {})
            correct_bool = bool(tc.get("correct", False))

            x_level = bin10_to_level(int(x_bins[j]))
            cell = (y_level, x_level)

            global_grid[cell]["total"] += 1
            per_question[qid][cell]["total"] += 1
            if correct_bool:
                global_grid[cell]["correct_true"] += 1
                per_question[qid][cell]["correct_true"] += 1
            else:
                global_grid[cell]["correct_false"] += 1
                per_question[qid][cell]["correct_false"] += 1

    return dict(global_grid), {qid: dict(cells) for qid, cells in per_question.items()}


def save_grid_image(
    ratio_grid: Dict[Tuple[int, int], float],
    count_grid: Dict[Tuple[int, int], Dict[str, int]],
    out_path: str,
    title: str = "",
    annotate: bool = True,
):
    """
    3x3 (Y level rows, X level cols) 이미지 저장.
    - ratio_grid[(y_level,x_level)] = correct_true/total
    - count_grid[(y_level,x_level)] = {"total":..., "correct_true":..., ...} (annotation용)
    """

    # 내부 레벨 인덱스: 0=Hard, 1=Medium, 2=Easy
    M = [[0.0 for _ in range(3)] for _ in range(3)]
    T = [[0 for _ in range(3)] for _ in range(3)]
    C = [[0 for _ in range(3)] for _ in range(3)]

    for (yb, xb), acc in ratio_grid.items():
        if 0 <= yb <= 2 and 0 <= xb <= 2:
            M[yb][xb] = float(acc)

    for (yb, xb), v in count_grid.items():
        if 0 <= yb <= 2 and 0 <= xb <= 2:
            T[yb][xb] = int(v.get("total", 0))
            C[yb][xb] = int(v.get("correct_true", 0))

    # 표시 순서를 Easy -> Medium -> Hard 로 바꿈
    order = [2, 1, 0]  # display_index -> original_level_index
    M_disp = [[M[order[y]][order[x]] for x in range(3)] for y in range(3)]
    T_disp = [[T[order[y]][order[x]] for x in range(3)] for y in range(3)]
    C_disp = [[C[order[y]][order[x]] for x in range(3)] for y in range(3)]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        M_disp,
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average Accuracy")

    x_labels = [
        "Easy (0.9-1.0)",
        "Medium (0.1-0.9)",
        "Hard (0.0-0.1)",
    ]
    y_labels = [
        "Low (0.0)",
        "Medium (0.0 < x < 1.0)",
        "High (1.0)",
    ]

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels, rotation=90, va="center")

    ax.set_xlabel("Testcase Difficulty")
    ax.set_ylabel("Code Quality")

    if annotate:
        for y in range(3):
            for x in range(3):
                ax.text(
                    x, y,
                    f"{M_disp[y][x]*100:.1f}%\n({C_disp[y][x]}/{T_disp[y][x]})",
                    ha="center", va="center", fontsize=9
                )


    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)



def finalize_ratio_maps(
    global_grid: Dict[Tuple[int, int], Dict[str, int]],
    per_question: Dict[str, Dict[Tuple[int, int], Dict[str, int]]],
):
    """
    count hashmap -> ratio hashmap(true/total)
    """
    global_ratio = {}
    for cell, v in global_grid.items():
        total = int(v.get("total", 0))
        ct = int(v.get("correct_true", 0))
        global_ratio[cell] = (ct / total) if total > 0 else 0.0

    per_question_ratio = {}
    for qid, grid in per_question.items():
        per_question_ratio[qid] = {}
        for cell, v in grid.items():
            total = int(v.get("total", 0))
            ct = int(v.get("correct_true", 0))
            per_question_ratio[qid][cell] = (ct / total) if total > 0 else 0.0

    return global_ratio, per_question_ratio


def get_hashmap(samples_json, results_jsonl, out_dir="result/hashmap"):
    os.makedirs(out_dir, exist_ok=True)

    samples = load_samples_json(samples_json)
    axes_map = build_axes_map(samples)

    global_grid, per_question = aggregate_correct_by_bins(axes_map, results_jsonl)

    global_ratio, per_question_ratio = finalize_ratio_maps(global_grid, per_question)

    with open(os.path.join(out_dir, "axes_by_question.json"), "w", encoding="utf-8") as f:
        json.dump(axes_map, f, ensure_ascii=False, indent=2)

    save_grid_image(
        ratio_grid=global_ratio,
        count_grid=global_grid,
        out_path=os.path.join(out_dir, "global_grid_ratio_3levels.png"),
        title="Global grid (Hard/Medium/Easy; cell=correct_true/total)",
        annotate=True,
    )

    print("Saved outputs to:", out_dir)