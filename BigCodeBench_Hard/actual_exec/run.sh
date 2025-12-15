#!/bin/bash

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Run generation for nucleus (temp 0.7) with 10 samples per problem
# echo "Running generation for nucleus (10 samples)..."
# python3 generate.py --models qwen3-coder-30B-A3B-instruct --sampling nucleus --n_samples 10

# Run evaluation for nucleus
echo "Running evaluation for nucleus (qwen3-coder-30B-A3B-instruct)..."
python3 run_eval.py --results_root results --models qwen3-coder-30B-A3B-instruct --sampling nucleus

echo "Evaluation complete. Check results/qwen3-coder-30B-A3B-instruct/nucleus_eval.json"
