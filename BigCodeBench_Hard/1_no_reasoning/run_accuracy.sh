#!/bin/bash
# 1_no_reasoning/run_accuracy.sh
# Model: claude-haiku-4-5-20251001 / gpt-5-mini-2025-08-07 / qwen3-coder-30B-A3B-instruct
# Default arguments for No_Reasoning
PRED_FILE=${1:-"results/qwen3-coder-30B-A3B-instruct/test.jsonl"}
MODEL_NAME=${2:-"qwen3-coder-30B-A3B-instruct"}

# Call the shared script
bash ../lib/run_accuracy.sh "$PRED_FILE" "$MODEL_NAME"
