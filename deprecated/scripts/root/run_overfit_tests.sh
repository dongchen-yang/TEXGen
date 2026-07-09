#!/bin/zsh

# Run all 3 baseline overfitting tests consecutively

# 1. Vanilla baseline
python launch.py --config configs/lightgen_pointuv_overfit.yaml --gpu 0 --train --wandb

# 2. GT mask conditioning
python launch.py --config configs/lightgen_pointuv_overfit_gt_mask_cond.yaml --gpu 0 --train --wandb

# 3. Mask classification loss
python launch.py --config configs/lightgen_pointuv_overfit_mask_cls.yaml --gpu 0 --train --wandb
