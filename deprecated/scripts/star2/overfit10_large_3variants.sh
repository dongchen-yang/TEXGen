#!/bin/bash
# Launch 3 overfit10 large-model variants on star2 (cs-venus-16)
# Variants: vanilla, gt_mask_cond, mask_cls_lambda001
# Architecture: 5-stage [32, 256, 1024, 1024, 2048]

CONFIGS=(
  "configs/lightgen_pointuv_overfit10_vanilla_large.yaml"
  "configs/lightgen_pointuv_overfit10_gt_mask_cond_large.yaml"
  "configs/lightgen_pointuv_overfit10_mask_cls_lambda001_large.yaml"
)
NAMES=(
  "overfit10_vanilla_large"
  "overfit10_gt_mask_cond_large"
  "overfit10_mask_cls_lambda001_large"
)

for i in "${!CONFIGS[@]}"; do
  CONFIG="${CONFIGS[$i]}"
  NAME="${NAMES[$i]}"

  echo "Submitting job: ${NAME} with config: ${CONFIG}"

  ssh star2 "sbatch" << EOF
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=5-15:00
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=cs-venus-16
#SBATCH --output=/localscratch/dya78/${NAME}.out
#SBATCH -J ${NAME}

echo "Start: ${NAME}"
echo "Config: ${CONFIG}"

source ~/.zshrc
cd /localscratch/dya78/lightgen/TEXGen
conda activate texgen

echo "pwd: \$(pwd)"
echo "GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python launch.py --config ${CONFIG} --gpu 0 --train --wandb

echo "Job End: ${NAME}"
EOF

  echo "Submitted: ${NAME}"
done

echo "All 3 jobs submitted."
