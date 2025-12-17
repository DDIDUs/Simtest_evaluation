#!/bin/bash

# Run Accuracy Calculation
# Compares predictions with ground truth.
# Model: qwen3-coder-30B-A3B-instruct / gpt-5-mini-2025-08-07 / claude-haiku-4-5-20251001
# Output saved to results/qwen3-coder-30B-A3B-instruct/accuracy_report.txt

python3 calc_accuracy.py \
    --pred_file results/claude-haiku-4-5-20251001/test.jsonl \
    --truth_file ../actual_exec/results/qwen3-coder-30B-A3B-instruct/nucleus_eval_all.json
