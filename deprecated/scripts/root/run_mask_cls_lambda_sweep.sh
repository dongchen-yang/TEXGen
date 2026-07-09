#!/bin/bash
# Mask classification loss lambda sweep: 0.01, 0.1, 10.0
# All runs on GPU 0 in parallel (~1.9GB each, ~5.7GB total on 24GB RTX 4090)
# Existing lambda=1.0 result already in outputs/overfit_mask_cls/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting mask_cls lambda sweep (0.01, 0.1, 10.0)..."

python launch.py --config configs/lightgen_pointuv_overfit_mask_cls_lambda001.yaml --gpu 0 --train &
PID1=$!
echo "  lambda=0.01  PID=$PID1"

python launch.py --config configs/lightgen_pointuv_overfit_mask_cls_lambda01.yaml --gpu 0 --train &
PID2=$!
echo "  lambda=0.1   PID=$PID2"

python launch.py --config configs/lightgen_pointuv_overfit_mask_cls_lambda10.yaml --gpu 0 --train &
PID3=$!
echo "  lambda=10.0  PID=$PID3"

echo ""
echo "All 3 runs launched. Waiting for completion..."
wait $PID1 $PID2 $PID3
echo "All runs finished."
