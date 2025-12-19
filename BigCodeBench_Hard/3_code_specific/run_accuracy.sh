#!/bin/bash
# 3_code_specific/run_accuracy.sh
# Model: claude-haiku-4-5-20251001 / gpt-5-mini-2025-08-07 / qwen3-coder-30B-A3B-instruct

# Default arguments for Code_Specific
PRED_FILE=${1:-"results/gpt-5-mini-2025-08-07/test.jsonl"}
MODEL_NAME=${2:-"gpt-5-mini-2025-08-07"}

# Call the shared script
bash ../lib/run_accuracy.sh "$PRED_FILE" "$MODEL_NAME"
