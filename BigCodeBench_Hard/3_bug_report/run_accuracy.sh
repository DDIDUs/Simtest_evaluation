#!/bin/bash

# Run Accuracy Calculation for 3_bug_report
# Compares predictions with ground truth.
# Output saved to results/qwen3-coder-30B-A3B-instruct/accuracy_report.txt

python3 calc_accuracy.py \
    --pred_file results/qwen3-coder-30B-A3B-instruct/test.jsonl \
    --truth_file ../actual_exec/results/qwen3-coder-30B-A3B-instruct/nucleus_eval_all.json
