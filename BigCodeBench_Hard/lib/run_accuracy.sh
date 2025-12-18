# Run Accuracy Calculation
# Usage: ./run_accuracy.sh <prediction_file> <model_name>
# Example: ./run_accuracy.sh results/qwen3-coder-30B-A3B-instruct/test.jsonl qwen3-coder-30B-A3B-instruct

PRED_FILE=${1:-"results/qwen3-coder-30B-A3B-instruct/test.jsonl"}
MODEL_NAME=${2:-"qwen3-coder-30B-A3B-instruct"}

# Assuming this script is called from inside lib/ or one level deep directory using ../lib path
# But actually the user plan says "Update to call ../lib/calc_accuracy.py". 
# So this script will be in lib/ and called by other scripts, OR this script is the template?
# Let's make this script flexible. It calculates accuracy relative to the current directory's results.

echo "Calculating accuracy for $PRED_FILE..."

# Calculate basic pass/fail accuracy
python3 ../lib/calc_accuracy.py --file "$PRED_FILE"

# Generate Heatmap (Task Pass Rate vs Testcase Pass Rate)
python3 ../lib/heatmap.py \
    --file "$PRED_FILE" \
    --output_dir "$(dirname "$PRED_FILE")/correlation" \
    --title "$MODEL_NAME Accuracy Heatmap"

# Generate 3-Level Binning Heatmap (High/Medium/Low vs Hard/Medium/Easy)
python3 ../lib/heatmap_3_level.py \
    --file "$PRED_FILE" \
    --output_dir "$(dirname "$PRED_FILE")/correlation" \
    --title "$MODEL_NAME 3-Level Accuracy Heatmap"
