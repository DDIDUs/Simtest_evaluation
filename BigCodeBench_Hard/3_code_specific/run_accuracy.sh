#!/bin/bash
# 3_code_specific/run_accuracy.sh

# Default arguments for Code_Specific
PRED_FILE=${1:-"results/qwen3-coder-30B-A3B-instruct/test.jsonl"}
MODEL_NAME=${2:-"Code_Specific"}

# Call the shared script
bash ../lib/run_accuracy.sh "$PRED_FILE" "$MODEL_NAME"
