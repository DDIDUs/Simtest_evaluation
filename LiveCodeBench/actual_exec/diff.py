import json
import os
import collections
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Paths (원래처럼 맞춰서 쓰세요)
RESULTS_DIR = Path("./out")
INPUT_FILE = Path("filtered_output.json")
OUTPUT_DIR = Path("./out")

def main():
    print("Starting classification script...", flush=True)
    if not INPUT_FILE.exists():
        print(f"Error: Input file NOT found at {INPUT_FILE}", flush=True)
        return

    print(f"Loading JSON from {INPUT_FILE}...", flush=True)
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # Map: task_id -> num_test_cases (tc 길이)
    task_num_tc = {}
    # Stats: unique_id -> {pass, total}
    tc_stats = {}

    def ensure_registry_for_task(task_id: str, num_tc: int):
        """task에 대해 tc_0..tc_{num_tc-1} 레지스트리 생성(이미 있으면 불일치 체크)"""
        if task_id in task_num_tc:
            if task_num_tc[task_id] != num_tc:
                # 같은 task_id에 대해 테스트케이스 수가 달라지는 경우는 데이터 이상 신호
                print(
                    f"Warning: Task {task_id} has inconsistent #testcases: "
                    f"{task_num_tc[task_id]} vs {num_tc}. Using the first seen value.",
                    flush=True
                )
            return

        task_num_tc[task_id] = num_tc
        for tc_idx in range(num_tc):
            unique_id = f"{task_id}#tc_{tc_idx}"
            tc_stats[unique_id] = {"pass": 0, "total": 0}

    processed_tasks = 0
    processed_samples = 0
    skipped_no_eval = 0
    skipped_bad_shape = 0

    print("Processing items...", flush=True)
    for item in tqdm(data, desc="Processing"):
        task_id = item.get("question_id") or item.get("task_id")
        if not task_id:
            continue

        eval_result = item.get("eval_result", None)
        if eval_result is None:
            skipped_no_eval += 1
            continue

        # eval_result: List[List[bool]] (샘플 x 테스트케이스)
        if not isinstance(eval_result, list) or len(eval_result) == 0:
            skipped_bad_shape += 1
            continue

        # 샘플들 중 첫 샘플로 tc 개수 추정
        first_row = eval_result[0]
        if not isinstance(first_row, list) or len(first_row) == 0:
            skipped_bad_shape += 1
            continue

        num_tc = len(first_row)
        ensure_registry_for_task(task_id, num_tc)

        # 샘플별 누적
        for row in eval_result:
            # row 길이가 num_tc와 다르면 데이터 이상이므로 안전하게 스킵(또는 min으로 자르기)
            if not isinstance(row, list) or len(row) != num_tc:
                skipped_bad_shape += 1
                continue

            processed_samples += 1

            # 이 샘플은 "모든 테스트케이스를 실행"한 결과라고 했으므로
            # total은 모든 tc에 대해 +1
            for tc_idx in range(num_tc):
                uid = f"{task_id}#tc_{tc_idx}"
                tc_stats[uid]["total"] += 1

                # row[tc_idx] == True 이면 pass
                if bool(row[tc_idx]):
                    tc_stats[uid]["pass"] += 1

        processed_tasks += 1

    print(f"Processed tasks: {processed_tasks}", flush=True)
    print(f"Processed samples: {processed_samples}", flush=True)
    print(f"Skipped (no eval_result): {skipped_no_eval}", flush=True)
    print(f"Skipped (bad shape): {skipped_bad_shape}", flush=True)

    # 3) Classify
    easy_tcs, medium_tcs, hard_tcs = [], [], []
    all_pass_rates = []

    for unique_id, stats in tc_stats.items():
        p = stats["pass"]
        t = stats["total"]
        rate = 0.0 if t == 0 else (p / t)
        all_pass_rates.append(rate)

        if rate == 1.0:
            easy_tcs.append(unique_id)
        elif rate == 0.0:
            hard_tcs.append(unique_id)
        else:
            medium_tcs.append(unique_id)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(all_pass_rates, bins=11, range=(-0.05, 1.05),
             alpha=0.6, color='skyblue', edgecolor='black', label='Test Case Count')
    plt.title(f'Test Case Pass Rate Distribution (N={len(all_pass_rates)})')
    plt.xlabel('Pass Rate')
    plt.ylabel('Number of Test Cases')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='Thresholds (10%, 90%)')
    plt.axvline(x=0.9, color='red', linestyle='--', linewidth=2)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plot_path = OUTPUT_DIR / "tc_pass_rate_distribution.png"
    plt.savefig(plot_path)
    print(f"Distribution plot saved to: {plot_path}", flush=True)

    # JSON save
    def save_category(name, tc_list):
        path = OUTPUT_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump({"count": len(tc_list), "ids": tc_list}, f, indent=2)

    save_category("easy", easy_tcs)
    save_category("medium", medium_tcs)
    save_category("hard", hard_tcs)

    print("-" * 30, flush=True)
    print("TC Level Classification Complete:", flush=True)
    print(f"Easy (100% pass): {len(easy_tcs)}", flush=True)
    print(f"Medium (Mixed):   {len(medium_tcs)}", flush=True)
    print(f"Hard (0% pass):   {len(hard_tcs)}", flush=True)
    print(f"Total Test Cases: {len(all_pass_rates)}", flush=True)
    print(f"Saved to {OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
