#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH -J lightgen_74k
#SBATCH --output=logs/lightgen_74k_%j.out
#SBATCH --open-mode=append
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=256G
#SBATCH --account=rrg-msavva
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@300

# Graceful timeout: when 5 min remain on the wall clock, kill training so it
# saves last.ckpt, then requeue. Lightning + auto_resume picks up on next start.
handle_timeout() {
    echo "[trap] timeout signal received (~5 min remaining); stopping training..."
    if [ -n "${TRAIN_PID:-}" ] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        kill -TERM "${TRAIN_PID}" 2>/dev/null
        wait "${TRAIN_PID}"
    fi
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap handle_timeout SIGTERM

# LightGen 74k training on fir.
#
# Usage (from ~/scratch/lightgen/TEXGen):
#   sbatch slurm_train_74k.sh
#   sbatch slurm_train_74k.sh CONFIG=configs/lightgen_pointuv_256_batch32_emissive_74k_mask_cls.yaml
#
# One-time prereqs (do once on the fir login node):
#   1. git pull in ~/scratch/lightgen/ to bring this script + new config + new split JSON
#   2. cp /cs/<jupiter or wherever>/df_SomgProc_final.parquet ~/scratch/lightgen/data/
#      (854k-row parquet is needed by the dataloader; ~373 MB)
#      If shipping from local workstation:
#        scp /local-scratch/.../data/baked_uv/df_SomgProc_final.parquet \
#            fir:~/scratch/lightgen/data/
#   3. Verify: ls ~/scratch/lightgen/data/df_SomgProc_final.parquet
#              ls ~/scratch/lightgen/data_processing/annotation/data_splits_emissive_74k_pinned.json
#              ls ~/scratch/lightgen/data/tars/npz_chunk_*.tar    # 8 files, ~57G each

set -euo pipefail

CONFIG=${CONFIG:-configs/lightgen_pointuv_256_batch32_emissive_74k.yaml}
TARS_DIR=${TARS_DIR:-/home/dya78/scratch/lightgen/data/tars}
PROJECT_ROOT=/home/dya78/scratch/lightgen

echo "============================================"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Node:           ${SLURM_NODELIST}"
echo "GPU:            ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_TMPDIR:   ${SLURM_TMPDIR}"
echo "Config:         ${CONFIG}"
echo "Start:          $(date -Iseconds)"
echo "============================================"

# Module / env setup. Module order matters: load native libs BEFORE activating
# the venv so wheels link against them (opencv, arrow, sparsehash).
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

# Stage NPZ data into the per-node SSD ($SLURM_TMPDIR).
# 8 tars × ~57 GB = ~490 GB extracted into $SLURM_TMPDIR/baked_uv/.
DATA_ROOT="${SLURM_TMPDIR}/baked_uv"
mkdir -p "${DATA_ROOT}"

echo ""
echo "[stage] checking node-local space:"
df -h "${SLURM_TMPDIR}" || true

echo ""
echo "[stage] extracting 8 NPZ tars from ${TARS_DIR} → ${DATA_ROOT} (parallel-of-8, with progress bar)"
T0=$(date +%s)
python "${PROJECT_ROOT}/TEXGen/detar_progress.py" "${TARS_DIR}" "${DATA_ROOT}"
T1=$(date +%s)
echo "[stage] extraction done in $((T1-T0))s"
echo "[stage] sample count:"
find "${DATA_ROOT}" -name 'somage.npz' | wc -l
echo "[stage] disk usage:"
du -sh "${DATA_ROOT}"

# Stage thumbnails for CLIP image conditioning. The dataloader looks at
# data_root/thumbnails/<id>.png; without these it falls back to the albedo UV
# map (substantively wrong conditioning input). Tar contains
# emissive_thumbnails/<id>.png; --strip-components=1 lands files directly
# under thumbnails/ to match the dataloader path with no symlink.
echo "[stage] extracting thumbnail tar (~12 GB)"
mkdir -p "${DATA_ROOT}/thumbnails"
tar -xf "${TARS_DIR}/thumbnails_emissive.tar" -C "${DATA_ROOT}/thumbnails" --strip-components=1
echo "[stage] thumbnails: $(find "${DATA_ROOT}/thumbnails/" -name '*.png' | wc -l) PNGs available"

# Verify dataloader prerequisites.
PARQUET="${PROJECT_ROOT}/data/df_SomgProc_final.parquet"
SPLITS="${PROJECT_ROOT}/data_processing/annotation/data_splits_emissive_74k_pinned.json"
for f in "${PARQUET}" "${SPLITS}"; do
    [ -f "$f" ] || { echo "MISSING: $f — see header for one-time setup"; exit 2; }
done

cd "${PROJECT_ROOT}/TEXGen"
mkdir -p logs

# Wandb (uses ~/.netrc for credentials; offline if missing).
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then
    export WANDB_MODE=offline
    echo "[wandb] no credentials — running offline (sync later with: wandb sync ...)"
fi

echo ""
echo "[train] launching"
echo "[train] python: $(which python)"
echo "[train] cuda:   $(python -c 'import torch; print(torch.version.cuda, torch.cuda.is_available())')"

# Override data_root to the runtime SLURM_TMPDIR via OmegaConf CLI.
# auto_resume in the config picks up last.ckpt on requeue.
# Run in background so the SIGTERM trap can fire and clean up.
python launch.py \
    --config "${CONFIG}" \
    --gpu 0 \
    --train \
    --wandb \
    "data.data_root=${DATA_ROOT}" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
EXIT_CODE=$?

# If training crashed (non-zero, not requeued via trap), requeue once.
if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "[main] training exit=${EXIT_CODE}; requeueing job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi
echo ""
echo "============================================"
echo "End:            $(date -Iseconds)"
echo "Exit code:      ${EXIT_CODE}"
echo "============================================"
exit ${EXIT_CODE}
