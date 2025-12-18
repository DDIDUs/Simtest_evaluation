#!/bin/bash

# Run BigCodeBench Hard Evaluation
# Model: qwen3-coder-30B-A3B-instruct / gpt-5-mini-2025-08-07 / claude-haiku-4-5-20251001
# Input: nucleus_code_generate.json (contains 10 samples per problem)
# Logic: Randomly selects 1 sample per problem and evaluates against test cases.
# Output: results/qwen3-coder-30B-A3B-instruct/test.jsonl (and test_raw.jsonl)

python3 run_eval.py \
    --generated_code ../actual_exec/results/qwen3-coder-30B-A3B-instruct/nucleus_code_generate.json \
    --model qwen3-coder-30B-A3B-instruct \
    --output_file test.jsonl
