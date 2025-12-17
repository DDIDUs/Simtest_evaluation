import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.eval_utils import *
from src.hashmap import get_hashmap

def evaluate(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    level_map: Dict[Tuple[str, str], str],
    target_code_idx: int,
    starts_at_0: bool,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    pred_map = index_by_question_id(preds)
    gt_map = index_by_question_id(gts)
    all_qids = sorted(set(pred_map.keys()) | set(gt_map.keys()))

    total_tasks = correct_tasks = 0
    total_tcs = correct_tcs = 0
    missing_pred = missing_gt = 0

    level_stats: Dict[str, Dict[str, int]] = {}
    results: List[Dict[str, Any]] = []

    for qid in all_qids:
        pred_row = pred_map.get(qid)
        gt_row = gt_map.get(qid)

        if pred_row is None:
            missing_pred += 1
        if gt_row is None:
            missing_gt += 1

        if pred_row is None or gt_row is None:
            total_tasks += 1
            update_level_stats(level_stats, "Unknown", task_total=1)
            results.append({"question_id": qid, "error": "missing_pred_or_gt"})
            continue

        total_tasks += 1

        gt_tc_bools = get_gt_tc_bools(gt_row, target_code_idx)
        if gt_tc_bools is None:
            update_level_stats(level_stats, "Unknown", task_total=1)
            results.append({"question_id": qid, "error": "gt_eval_result_invalid"})
            continue

        pred_code = find_pred_code(pred_row, target_code_idx)
        if pred_code is None:
            update_level_stats(level_stats, "Unknown", task_total=1)
            results.append({"question_id": qid, "error": "pred_code_index_missing"})
            continue

        pred_overall = bool(pred_code.get("overall_pass", False))
        pred_pf_list = pred_code.get("pass_fail_list", [])
        if not isinstance(pred_pf_list, list):
            pred_pf_list = []

        n = decide_num_testcases(pred_row, gt_tc_bools, pred_pf_list)
        gt_tc_bools = pad_or_trim(gt_tc_bools, n, None)
        pred_pf_list = pad_or_trim(pred_pf_list, n, None)

        gt_overall = compute_gt_overall(gt_tc_bools)
        task_level = infer_task_level(qid, n, level_map, starts_at_0)
        update_level_stats(level_stats, task_level, task_total=1)

        overall_correct = (pred_overall == gt_overall)
        if overall_correct:
            correct_tasks += 1
            update_level_stats(level_stats, task_level, task_correct=1)

        tc_details: Dict[str, Any] = {}
        for i in range(n):
            name = tc_name(i, starts_at_0)
            level = level_map.get((qid, name), "Unknown")

            gt_bool = gt_tc_bools[i]
            gt_status = None if gt_bool is None else ("PASS" if gt_bool else "FAIL")

            from src.eval_utils import normalize_pred_passfail  # keep local to avoid circulars if any
            pred_bool = normalize_pred_passfail(pred_pf_list[i])
            pred_status = "PASS" if pred_bool else "FAIL"

            total_tcs += 1
            update_level_stats(level_stats, level, tc_total=1)

            correct = (pred_status == gt_status)
            if correct:
                correct_tcs += 1
                update_level_stats(level_stats, level, tc_correct=1)

            tc_details[name] = {
                "level": level,
                "gt": gt_status,
                "pred": pred_status,
                "correct": correct,
            }

        results.append(
            {
                "question_id": qid,
                "overall_correct": overall_correct,
                "gt_overall": "PASS" if gt_overall else "FAIL",
                "pred_overall": "PASS" if pred_overall else "FAIL",
                "num_testcases_used": n,
                "test_cases": tc_details,
            }
        )

    report = build_report(
        total_tasks=total_tasks,
        correct_tasks=correct_tasks,
        total_tcs=total_tcs,
        correct_tcs=correct_tcs,
        level_stats=level_stats,
        missing_pred=missing_pred,
        missing_gt=missing_gt,
    )

    summary = {
        "total_tasks": total_tasks,
        "correct_tasks": correct_tasks,
        "total_tcs": total_tcs,
        "correct_tcs": correct_tcs,
        "missing_pred": missing_pred,
        "missing_gt": missing_gt,
    }
    return report, results, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="pred.json 또는 pred.jsonl")
    parser.add_argument("--gt_file", type=str, required=True, help="gt.json 또는 gt.jsonl")
    parser.add_argument("--level_dir", type=str, default='../actual_exec/out/test_distribution')
    parser.add_argument("--target_code_idx", type=int, default=0)
    parser.add_argument("--level_tc_starts_at_0", action="store_false")
    args = parser.parse_args()

    preds = load_jsonl_or_json(args.pred_file)
    gts = load_jsonl_or_json(args.gt_file)

    level_map: Dict[Tuple[str, str], str] = {}
    if args.level_dir:
        level_map = build_level_map(args.level_dir)

    report, results, _ = evaluate(
        preds=preds,
        gts=gts,
        level_map=level_map,
        target_code_idx=args.target_code_idx,
        starts_at_0=args.level_tc_starts_at_0,
    )

    print(report)

    out_dir = Path(args.pred_file).parent
    save_outputs(out_dir, report, results)

    print(f"\nSaved: {out_dir / 'accuracy_report.txt'}")
    print(f"Saved: {out_dir / 'accuracy_raw.jsonl'}")

    get_hashmap(Path(args.gt_file), {out_dir / "accuracy_raw.jsonl"}.pop(), Path(out_dir))


if __name__ == "__main__":
    main()
