#!/bin/bash
# Launch mask_cls_semantic training on fir cluster
# Config: lightgen_pointuv_256_batch32_emission_filtered_mask_cls.yaml

CONFIG="configs/lightgen_pointuv_256_batch32_emission_filtered_mask_cls.yaml"
NAME="e_mask_condition"

echo "Submitting job: ${NAME} with config: ${CONFIG}"

ssh fir "sbatch" << EOF
#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=%N-%j.out
#SBATCH --open-mode=append
#SBATCH --mem=256G
#SBATCH --account=rrg-msavva
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@300

handle_timeout() {
    echo "Timeout signal received (5 min remaining). Stopping training gracefully..."
    if [ ! -z "\$TRAIN_PID" ] && kill -0 \$TRAIN_PID 2>/dev/null; then
        kill -TERM \$TRAIN_PID 2>/dev/null
        echo "Waiting for training to save checkpoint..."
        wait \$TRAIN_PID
        echo "Training process exited. Requeueing job..."
    fi
    scontrol requeue \${SLURM_JOB_ID}
    exit 0
}

trap 'handle_timeout' SIGTERM

echo "======================================"
echo "Job: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})"
echo "Started: \$(date)"
echo "======================================"

module load StdEnv/2023 cuda/12.6
module load gcc
module load opencv
module load arrow/22.0.0
module load sparsehash

source /scratch/dya78/lightgen/env/bin/activate

cd \$SLURM_TMPDIR
git clone git@github.com:dongchen-yang/lightgen.git --recursive
cd lightgen/

./unpack_texgen_data.sh

cd TEXGen

git pull origin main

# Run training in background so SIGTERM trap can fire while waiting
python launch.py --config ${CONFIG} --gpu 0 --train --wandb &
TRAIN_PID=\$!
wait \$TRAIN_PID
EXIT_CODE=\$?

if [ \$EXIT_CODE -ne 0 ]; then
    echo "Training crashed (exit code \$EXIT_CODE) at \$(date). Requeueing job..."
    scontrol requeue \${SLURM_JOB_ID}
    exit 0
fi

echo "Training completed at \$(date)."
EOF

echo "Submitted: ${NAME}"
