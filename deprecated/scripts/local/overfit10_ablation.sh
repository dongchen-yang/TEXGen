#!/bin/bash
# Ablation study: leave-one-out on vanilla allreg (10-sample overfit)
# 4 runs: remove weight_decay / EMA / dropout / condition_dropout one at a time
# Parallel runs (~2GB VRAM each, all fit on one GPU)

cd /localhome/dya78/code/lightgen/TEXGen

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting ablation study (4 runs, parallel on GPU 0)..."

echo "=== [1/4] allreg - no weight decay ==="
python launch.py --config configs/lightgen_pointuv_overfit10_vanilla_allreg_no_wd.yaml --gpu 0 --train --wandb &
PID1=$!

echo "=== [2/4] allreg - no EMA ==="
python launch.py --config configs/lightgen_pointuv_overfit10_vanilla_allreg_no_ema.yaml --gpu 0 --train --wandb &
PID2=$!

echo "=== [3/4] allreg - no dropout ==="
python launch.py --config configs/lightgen_pointuv_overfit10_vanilla_allreg_no_dropout.yaml --gpu 0 --train --wandb &
PID3=$!

echo "=== [4/4] allreg - no condition dropout ==="
python launch.py --config configs/lightgen_pointuv_overfit10_vanilla_allreg_no_conddrop.yaml --gpu 0 --train --wandb &
PID4=$!

echo "All 4 ablation runs launched (PIDs: $PID1, $PID2, $PID3, $PID4). Waiting..."
wait $PID1 $PID2 $PID3 $PID4
echo "All ablation runs finished."
