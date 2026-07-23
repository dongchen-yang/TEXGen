#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH -J texgen_74k_v2
#SBATCH --output=logs/texgen_74k_v2_%j.out
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=250G
#SBATCH --account=aip-msavva
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@300

# TEXGen vanilla _v2 on VULCAN — 2 nodes x 4 L40S, global batch 32
# (= the single-H100 fir reference: per-GPU batch 4 x 8 ranks).
#
# Mirrors slurm_train_74k.sh (fir) with the multi-node deltas:
#   * ntasks-per-node == devices (Lightning SLURM env asserts this).
#   * tar staging runs ON EVERY NODE (per-node $SLURM_TMPDIR).
#   * NO_PROXY bypass for NCCL rendezvous (vulcan httpproxy).
#   * trainer.num_nodes=2 baked into the _v2 config.
#
# One-time prereqs (on the vulcan login node):
#   1. clone lightgen to /scratch/dya78/lightgen (recursive, lightgen branch of
#      TEXGen submodule at the texgen-74k-v2-vulcan commit) + build env/ per
#      cluster-access fir/env-setup.md (same modules as below, --no-index wheels)
#   2. /scratch/dya78/lightgen/data/df_SomgProc_final.parquet
#      /scratch/dya78/lightgen/data/data_splits_emissive_74k_stratified_newbake.json
#   3. /scratch/dya78/lightgen/data/tars_v2/texgen_root_chunk_*.tar  (8 chunks)
#      /scratch/dya78/lightgen/data/tars_v2/thumbnails_emissive.tar
#
# Usage (from /scratch/dya78/lightgen/TEXGen):
#   sbatch scripts/vulcan/slurm_train_74k_v2.sh

set -euo pipefail

handle_timeout() {
    echo "[trap] timeout signal received (~5 min remaining); stopping training..."
    if [ -n "${TRAIN_PID:-}" ] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        kill -TERM "${TRAIN_PID}" 2>/dev/null
        wait "${TRAIN_PID}" || true   # nonzero exit is expected; don't let errexit skip the requeue
    fi
    scontrol requeue "${SLURM_JOB_ID}" || true
    exit 0
}
trap handle_timeout SIGTERM

CONFIG=${CONFIG:-configs/lightgen_pointuv_256_batch32_emissive_74k_v2.yaml}
TARS_DIR=${TARS_DIR:-/scratch/dya78/lightgen/data/tars_v2}
PROJECT_ROOT=/scratch/dya78/lightgen

echo "============================================"
echo "Job ID:    ${SLURM_JOB_ID}   Nodes: ${SLURM_NODELIST}"
echo "Config:    ${CONFIG}"
echo "Start:     $(date -Iseconds)"
echo "============================================"

# Multi-node NCCL rendezvous: bypass vulcan's httpproxy for intra-job traffic.
export NO_PROXY=localhost,127.0.0.1,$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
export no_proxy="$NO_PROXY"

# Module / env setup (same stack as fir; venv built per fir/env-setup.md).
module load StdEnv/2023 cuda/12.6
module load gcc
module load opencv
module load arrow/22.0.0
module load sparsehash
module load python/3.11.5
source "${PROJECT_ROOT}/env/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-${PROJECT_ROOT}/.cache/torch}
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

# Stage data on EVERY node's local SSD (per-node $SLURM_TMPDIR).
DATA_ROOT="${SLURM_TMPDIR}/texgen_train_root"
echo "[stage] extracting on all nodes (chunks + thumbnails)..."
T0=$(date +%s)
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --cpus-per-task=12 bash -c '
    set -e
    mkdir -p "'"${DATA_ROOT}"'"
    python "'"${PROJECT_ROOT}"'/TEXGen/detar_progress.py" "'"${TARS_DIR}"'" "'"${DATA_ROOT}"'" "texgen_root_chunk_*.tar"
    mkdir -p "'"${DATA_ROOT}"'/thumbnails"
    tar -xf "'"${TARS_DIR}"'/thumbnails_emissive.tar" -C "'"${DATA_ROOT}"'/thumbnails" --strip-components=1
    echo "[stage:$(hostname)] npz: $(find "'"${DATA_ROOT}"'" -name somage.npz | wc -l), thumbs: $(ls "'"${DATA_ROOT}"'/thumbnails" | wc -l)"
'
echo "[stage] all nodes done in $(( $(date +%s) - T0 ))s"

# Dataloader prerequisites.
for f in "${PROJECT_ROOT}/data/df_SomgProc_final.parquet" \
         "${PROJECT_ROOT}/data/data_splits_emissive_74k_stratified_newbake.json"; do
    [ -f "$f" ] || { echo "MISSING: $f — see header"; exit 2; }
done

cd "${PROJECT_ROOT}/TEXGen"
mkdir -p logs

if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then
    export WANDB_MODE=offline
    echo "[wandb] no credentials — running offline (sync later)"
fi

echo "[train] launching 2x4 L40S DDP (global batch 32)"
# srun spawns ntasks-per-node=4 tasks/node; Lightning's SLURM env consumes the
# ranks (config: trainer.num_nodes=2, devices=-1 -> 4 visible GPUs per task set).
# auto_resume picks up last.ckpt on requeue.
# --gpu-bind=none: every task must see all 4 local GPUs (Lightning picks by
# SLURM_LOCALID; per-task binding would collapse devices to 1 and trip the
# ntasks-per-node == devices assertion).
srun --gpu-bind=none python launch.py \
    --config "${CONFIG}" \
    --gpu 0 \
    --train \
    --wandb \
    "data.data_root=${DATA_ROOT}" &
TRAIN_PID=$!
EXIT_CODE=0
wait "${TRAIN_PID}" || EXIT_CODE=$?

echo "[done] train exit ${EXIT_CODE} at $(date -Iseconds)"
if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "[done] non-zero exit — requeueing once for transient failures"
    scontrol requeue "${SLURM_JOB_ID}" || true
fi
exit "${EXIT_CODE}"
