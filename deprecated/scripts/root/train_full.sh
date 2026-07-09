#!/bin/bash
# Training script for full annotated dataset

export CUDA_VISIBLE_DEVICES=0

# Activate conda environment
source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate texgen

# Run training
python train.py \
    --config configs/lightgen_full.yaml \
    --gpu 0 \
    --train \
    2>&1 | tee logs/train_full_$(date +%Y%m%d_%H%M%S).log



