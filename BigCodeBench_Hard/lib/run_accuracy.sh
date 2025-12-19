# Run Accuracy Calculation
# Usage: ./run_accuracy.sh <prediction_file> <model_name>
# Example: ./run_accuracy.sh results/qwen3-coder-30B-A3B-instruct/test.jsonl qwen3-coder-30B-A3B-instruct

PRED_FILE=${1:-"results/qwen3-coder-30B-A3B-instruct/test.jsonl"}
MODEL_NAME=${2:-"qwen3-coder-30B-A3B-instruct"}

# Determine relative path from current directory to root if possible, or assume calling location.
# Better strategy: Get absolute path of PRED_FILE to determine where we are.
PRED_ABS=$(realpath "$PRED_FILE")
CURRENT_DIR=$(pwd)
DIR_NAME=$(basename "$CURRENT_DIR")

# Assuming directory structure: BigCodeBench_Hard/<DIR_NAME>
# Root is ../
# Truth file is at ../actual_exec/results/qwen3-coder-30B-A3B-instruct/nucleus_eval_all.json
# Wait, Qwen's result is the TRUTH? Yes, based on python script comments "ALWAYS use Qwen's eval data".
TRUTH_FILE="../actual_exec/results/qwen3-coder-30B-A3B-instruct/nucleus_eval_all.json"

echo "Directory: $DIR_NAME"
echo "Calculating accuracy for $PRED_FILE..."
echo "Truth File: $TRUTH_FILE"

# Calculate basic pass/fail accuracy
python3 ../lib/calc_accuracy.py \
    --pred_file "$PRED_FILE" \
    --truth_file "$TRUTH_FILE"

# Generate 3-Level Binning Heatmap (High/Medium/Low vs Hard/Medium/Easy)
python3 ../lib/heatmap_3_level.py \
    --model "$MODEL_NAME" \
    --dir "$DIR_NAME"

