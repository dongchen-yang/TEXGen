#!/bin/bash
# Convenience script to start training with TensorBoard

set -e

echo "=========================================="
echo "LightGen Training with TensorBoard"
echo "=========================================="
echo ""
echo "Sample: 416f4870df6449dfaf9533be8aa18701"
echo "Config: configs/lightgen.yaml"
echo ""

# Check if in correct directory
if [ ! -f "launch.py" ]; then
    echo "Error: Please run this script from TEXGen directory"
    exit 1
fi

# Activate conda environment
echo "Activating texgen environment..."
eval "$(conda shell.bash hook)"
conda activate texgen

# Start TensorBoard in background
echo "Starting TensorBoard on port 6006..."
tensorboard --logdir outputs/lightgen/v1/tb_logs --port 6006 --host 0.0.0.0 > tensorboard.log 2>&1 &
TB_PID=$!
echo "TensorBoard PID: $TB_PID"
echo "TensorBoard logs: tensorboard.log"
echo ""

# Wait a moment for TensorBoard to start
sleep 2

# Print instructions
echo "=========================================="
echo "TensorBoard is running!"
echo "=========================================="
echo ""
echo "Access TensorBoard at:"
echo "  http://localhost:6006"
echo ""
echo "To stop TensorBoard later:"
echo "  kill $TB_PID"
echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

# Start training
python launch.py --config configs/lightgen.yaml --train --gpu 0

# Note: This script will keep running until training finishes or you press Ctrl+C
# TensorBoard will keep running in the background even after training stops
echo ""
echo "Training finished!"
echo ""
echo "TensorBoard is still running (PID: $TB_PID)"
echo "To stop it: kill $TB_PID"

