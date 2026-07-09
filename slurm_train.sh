#!/bin/bash
#SBATCH --job-name=lightgen_pointuv
#SBATCH --output=logs/lightgen_%j.out
#SBATCH --error=logs/lightgen_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# LightGen Training Script for Cluster
# Usage: sbatch slurm_train.sh

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================"

# CRITICAL: Prevent PyTorch memory fragmentation (fixes OOM issues)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Set wandb API key (if not using ~/.netrc)
# export WANDB_API_KEY="your_api_key_here"

# Optional: Use offline mode (sync later)
# export WANDB_MODE=offline

# HuggingFace token for accessing gated models (required for stable-diffusion-2-depth)
# Get your token from: https://huggingface.co/settings/tokens
# export HF_TOKEN="hf_your_token_here"
# Note: If you've run 'huggingface-cli login', the token is cached and this is optional

# Load conda
source ~/.bashrc
conda activate texgen

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Print environment info
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Verify wandb connection
echo "Verifying wandb connection..."
wandb verify || echo "Warning: wandb verification failed, continuing anyway..."
echo ""

# Train with wandb monitoring
echo "Starting training..."
python launch.py \
    --config configs/lightgen_pointuv_256_batch32_emission_filtered.yaml \
    --gpu 0 \
    --train \
    --wandb

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Training finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================"

# Optional: If using offline mode, sync to wandb
# if [ "$WANDB_MODE" = "offline" ]; then
#     echo "Syncing offline runs to wandb..."
#     wandb sync outputs/lightgen/pointuv_full@*/wandb/ || echo "Sync failed, sync manually later"
# fi

exit $EXIT_CODE



