#!/bin/zsh

# Pretrained
python launch.py --config configs/lightgen_pointuv_overfit_pretrained.yaml --gpu 0 --wandb --train
# Baseline
python launch.py --config configs/lightgen_pointuv_overfit.yaml --gpu 0 --wandb --train