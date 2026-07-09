#!/bin/bash
# 10-sample overfitting tests: vanilla, GT mask cond, mask cls (lambda=0.01)
# Parallel runs (~2GB VRAM each, all fit on one GPU)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting 10-sample overfit tests (3 variants, parallel on GPU 0)..."

echo "=== [1/3] Vanilla ==="
python launch.py --config configs/lightgen_pointuv_overfit10_vanilla.yaml --gpu 0 --train --wandb &
PID1=$!

echo "=== [2/3] GT Mask Cond ==="
python launch.py --config configs/lightgen_pointuv_overfit10_gt_mask_cond.yaml --gpu 0 --train --wandb &
PID2=$!

echo "=== [3/3] Mask Cls lambda=0.01 ==="
python launch.py --config configs/lightgen_pointuv_overfit10_mask_cls_lambda001.yaml --gpu 0 --train --wandb &
PID3=$!

echo "All 3 runs launched (PIDs: $PID1, $PID2, $PID3). Waiting..."
wait $PID1 $PID2 $PID3
echo "All runs finished."
