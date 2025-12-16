#!/bin/bash
set -e

# Image name
IMAGE_NAME="bigcodebench-eval-local"

echo "Building Docker image: $IMAGE_NAME..."
# Build the image using the Dockerfile in the current directory
docker build -t "$IMAGE_NAME" .

echo "Starting evaluation in Docker container..."
echo "Mounting $(pwd) to /app"

# Run the container
# --rm: Remove container after exit
# -v $(pwd):/app: Mount current directory to /app
# --shm-size=4g: Increase shared memory size (often needed for large multi-processing)
# --network none: (Optional) If you want to block internet access for extra safety, but might break dataset loading if not cached. 
#                 Official template doesn't strict output network, but we can verify.
#                 For now, we allow network as the script might need to load dataset/models if not present.
docker run --rm \
    -u "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$(pwd)":/app \
    --shm-size=4g \
    "$IMAGE_NAME" \
    python3 run_eval.py --results_root results --models qwen3-coder-30B-A3B-instruct --sampling nucleus

echo "Docker execution finished."
