#!/bin/bash
# 1_no_reasoning/run_accuracy.sh

# Default arguments for No_Reasoning
PRED_FILE=${1:-"results/qwen3-coder-30B-A3B-instruct/test.jsonl"}
MODEL_NAME=${2:-"No_Reasoning"}

# Call the shared script
bash ../lib/run_accuracy.sh "$PRED_FILE" "$MODEL_NAME"
