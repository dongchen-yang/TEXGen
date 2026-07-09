#!/bin/bash
# 10-sample overfitting tests with all regularization (EMA, dropout, weight_decay, cond_drop)
# 3 variants: Vanilla, GT Mask Cond, Mask Cls λ=0.01 — small (3-stage) architecture
# Parallel runs (~2GB VRAM each, all fit on one GPU)

cd /localhome/dya78/code/lightgen/TEXGen

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting 10-sample overfit allreg tests (3 variants, parallel on GPU 0)..."

echo "=== [1/3] Vanilla + allreg ==="
python launch.py --config configs/lightgen_pointuv_overfit10_vanilla_allreg.yaml --gpu 0 --train --wandb &
PID1=$!

echo "=== [2/3] GT Mask Cond + allreg ==="
python launch.py --config configs/lightgen_pointuv_overfit10_gt_mask_cond_allreg.yaml --gpu 0 --train --wandb &
PID2=$!

echo "=== [3/3] Mask Cls λ=0.01 + allreg ==="
python launch.py --config configs/lightgen_pointuv_overfit10_mask_cls_lambda001_allreg.yaml --gpu 0 --train --wandb &
PID3=$!

echo "All 3 runs launched (PIDs: $PID1, $PID2, $PID3). Waiting..."
wait $PID1 $PID2 $PID3
echo "All runs finished."
