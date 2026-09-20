#!/bin/bash -l
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH -J texgen_vanilla_74k_v2
#SBATCH --output=/scratch/dya78/lightgen_repo/logs/texgen_v2_1node_%j.out
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=250G
#SBATCH --account=aip-msavva
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@300

# TEXGen vanilla _v2 on vulcan — ONE node x 4 L40S, micro-batch 8 x
# accumulate_grad_batches 4 = GLOBAL BATCH 128 (the 4x H100 reference).
#
# Replaces scripts/vulcan/slurm_train_74k_v2.sh, which was 2 nodes x 4 L40S and
# never reached step 0: job 83679 requeued ~19 times between 2026-07-24 and 07-27,
# hanging every time in c10d::ProcessGroupNCCL::broadcastUniqueNCCLID on a 1800 s
# TCPStore timeout. That rendezvous is cross-node only, so staying on one node
# removes the failure mode entirely rather than working around it.
#
# Reaching global 128 on one node needs the accumulation trade: an L40S holds
# 44.39 GiB usable and per-GPU 32 needs far more. Micro-batch 16 was tried first
# and OOM'd at a measured 41.99 GB peak (job 217827) — the documented H100 sizing
# model underestimates on this stack. Micro-batch 8 x 4 is the working setting;
# see the measured table in TEXGen/CLAUDE.md.
#
# `#!/bin/bash -l` is load-bearing — a plain bash batch shell on vulcan has no
# `module` function, so the whole module stack silently no-ops and CUDA_HOME comes
# out empty.
#
# Usage (from /scratch/dya78/lightgen_repo/TEXGen):
#   sbatch scripts/vulcan/train_74k_v2_1node.sh
#   EXTRA="trainer.max_steps=30 trainer.limit_val_batches=2" sbatch --export=ALL,EXTRA scripts/vulcan/train_74k_v2_1node.sh

set -uo pipefail

NAME=texgen_vanilla_74k_v2
CONFIG=${CONFIG:-configs/lightgen_pointuv_256_batch32_emissive_74k_v2_vulcan1node.yaml}
EXTRA=${EXTRA:-}
# Which local GPUs Lightning sees. Default is the production 4; override to a single
# device for a cheap per-GPU VRAM probe, which schedules far sooner than a 4-GPU
# reservation and answers the only per-GPU question there is (VRAM is per device).
GPUS=${GPUS:-0,1,2,3}
PROJECT=/scratch/dya78/lightgen_repo
TARS_DIR=${TARS_DIR:-$PROJECT/data/tars_v2}
ATTEMPTS=$PROJECT/.texgen_v2_1node_attempts
MAX_ATTEMPTS=3

# Requeue ONLY on the wall-clock signal, plus a hard cap on crash requeues. The
# 2-node script requeued on ANY non-zero exit, which is what turned one persistent
# failure into ~19 silent retries appended into a single log.
handle_timeout() {
    echo "[trap] wall-clock signal (~5 min left); stopping for requeue..."
    if [ -n "${TRAIN_PID:-}" ] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        kill -TERM "${TRAIN_PID}" 2>/dev/null
        wait "${TRAIN_PID}" || true
    fi
    echo 0 > "$ATTEMPTS"
    scontrol requeue "${SLURM_JOB_ID}" || true
    exit 0
}
trap handle_timeout SIGTERM

echo "============================================"
echo "Job:    ${SLURM_JOB_ID}   Node: $(hostname)"
echo "Config: ${CONFIG}   GPUs: ${GPUS}   Extra: '${EXTRA}'"
echo "Start:  $(date -Iseconds)"
echo "============================================"

module load StdEnv/2023 cuda/12.6
module load gcc opencv arrow/22.0.0 sparsehash python/3.11.5
source "${PROJECT}/env/bin/activate"

# torch 2.9 renamed this; set both so the setting is honoured either way. It is not
# cosmetic — the OOM in job 217827 explicitly recommended expandable_segments, and
# the run had 42 GB live against 44.39 GiB usable, where fragmentation decides.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-${PROJECT}/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-${PROJECT}/.cache/torch}
export HF_HUB_OFFLINE=1        # compute nodes have no internet; CLIP is pre-cached
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${PROJECT}/logs"

# Single node, so Lightning must spawn its own 4 ranks rather than expect one task
# per rank from SLURM. Without this it uses the SLURM launcher, which asserts
# ntasks == devices, and one task with 4 devices hangs at "Initializing
# distributed" — which reads exactly like a NCCL fault and is not one.
export SLURM_JOB_NAME=bash

DATA_ROOT="${SLURM_TMPDIR}/texgen_train_root"
echo "[stage] extracting chunks + thumbnails to ${DATA_ROOT} ..."
T0=$(date +%s)
mkdir -p "${DATA_ROOT}/thumbnails"
python "${PROJECT}/TEXGen/detar_progress.py" "${TARS_DIR}" "${DATA_ROOT}" "texgen_root_chunk_*.tar" || exit 1
tar -xf "${TARS_DIR}/thumbnails_emissive.tar" -C "${DATA_ROOT}/thumbnails" --strip-components=1 || exit 1
echo "[stage] done in $(( $(date +%s) - T0 ))s"
echo "[stage] npz=$(find "${DATA_ROOT}" -name somage.npz | wc -l)  thumbs=$(ls "${DATA_ROOT}/thumbnails" | wc -l)  (expect 72421 / 72421)"

for f in "${PROJECT}/data/df_SomgProc_final.parquet" \
         "${PROJECT}/data/data_splits_emissive_74k_stratified_newbake_vae.json"; do
    [ -f "$f" ] || { echo "MISSING: $f"; exit 2; }
done

cd "${PROJECT}/TEXGen"
echo "TEXGen HEAD: $(git log --oneline -1)"

if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then
    export WANDB_MODE=offline
    echo "[wandb] no credentials — offline, sync later with: wandb sync -p LightGen <dir>"
fi

# No `| tee`: SLURM's --output already persists everything, and piping would make
# TRAIN_PID the tee process, so `wait` would report TEE's status and a crashed
# trainer would be read as success.
python launch.py --config "${CONFIG}" --gpu "${GPUS}" --train --wandb \
    "data.data_root=${DATA_ROOT}" ${EXTRA} &
TRAIN_PID=$!
wait "${TRAIN_PID}"
EXIT_CODE=$?

echo "[done] train exit ${EXIT_CODE} at $(date -Iseconds)"
if [ "${EXIT_CODE}" -ne 0 ]; then
    N=$(( $(cat "$ATTEMPTS" 2>/dev/null || echo 0) + 1 ))
    echo "$N" > "$ATTEMPTS"
    if [ "$N" -lt "$MAX_ATTEMPTS" ]; then
        echo "[done] crash ${N}/${MAX_ATTEMPTS} — requeueing"
        scontrol requeue "${SLURM_JOB_ID}" || true
    else
        echo "[done] crash ${N}/${MAX_ATTEMPTS} — NOT requeueing; inspect the log"
    fi
fi
exit "${EXIT_CODE}"
