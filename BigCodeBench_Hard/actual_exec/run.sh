#!/bin/bash

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Run generation for nucleus (temp 0.7) with 10 samples per problem
echo "Running generation for nucleus (10 samples)..."
python3 generate.py --models qwen3-coder-30B-A3B-instruct --sampling neuclus --n_samples 10

# Run evaluation for nucleus
echo "Running evaluation for nucleus..."
python3 run_eval.py --models qwen3-coder-30B-A3B-instruct --sampling neuclus

echo "Dryrun complete. Check results/ and log files."
