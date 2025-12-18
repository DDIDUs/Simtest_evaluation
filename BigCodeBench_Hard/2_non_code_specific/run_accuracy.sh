#!/bin/bash
# 2_non_code_specific/run_accuracy.sh

# Default arguments for Non_Code_Specific
PRED_FILE=${1:-"results/qwen3-coder-30B-A3B-instruct/test.jsonl"}
MODEL_NAME=${2:-"Non_Code_Specific"}

# Call the shared script
bash ../lib/run_accuracy.sh "$PRED_FILE" "$MODEL_NAME"
