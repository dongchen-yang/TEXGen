#!/bin/bash
# TEXGen vanilla _v2 on cs-venus-05 — 1 node x 4 RTX PRO 6000 Blackwell Max-Q,
# per-GPU batch 32 -> GLOBAL BATCH 128 (the 4x H100 reference).
#
# RUNS FROM THE WORKSTATION. It pipes the job body into `ssh star2 sbatch`; the
# script itself never needs to exist on the compute node. That is not a style
# choice: on star2 BOTH /localscratch and /home are per-node, so a path like
# /localscratch/dya78/lightgen does not exist on the head node and
# `ssh star2 'cd <that> && sbatch ...'` cannot work. Same pattern as
# TRELLIS2/script/star2/emission_dit_1k_probe_venus08.sh.
#
# Usage:
#   bash scripts/star2/train_74k_v2_venus05.sh                       # full run
#   EXTRA="trainer.max_steps=30 trainer.limit_val_batches=2" bash ...  # smoke
#   NODE=cs-venus-19 GRES=rtx_pro_6000_blackwell_se CPUS=32 bash ...   # fallback node
set -euo pipefail

NAME=texgen_alpha_74k_v2
CONFIG=${CONFIG:-configs/lightgen_pointuv_256_batch32_emissive_74k_v2_alpha_venus05.yaml}
EXTRA=${EXTRA:-}
# Which local GPUs Lightning sees. VRAM is a per-GPU quantity, so a 1-GPU probe
# answers the sizing question and schedules sooner than a 4-GPU reservation.
GPUS=${GPUS:-0,1,2,3}
NODE=${NODE:-cs-venus-05}
GRES=${GRES:-rtx_pro_6000_blackwell_max-q}
NGPU=${NGPU:-4}
CPUS=${CPUS:-48}
# 200G was not enough: job 237314 died OUT_OF_MEMORY with MaxRSS 209,710,500K, i.e.
# exactly the 200G ceiling. 28 processes (4 ranks x 1 main + 6 workers) each hold a
# copy of the 824,858-row sample index built from the parquet, and wandb online adds
# upload buffers on top. The node has 503 GB and is otherwise idle; 400G leaves ~100.
MEM=${MEM:-400G}
WALL=${WALL:-7-00:00:00}
PROJECT=/localscratch/dya78/lightgen
BRANCH=texgen-74k-v2-venus05

echo "Submitting ${NAME}"
echo "  node   : ${NODE}  (${NGPU} x ${GRES}, ${CPUS} cpu, ${MEM})"
echo "  config : ${CONFIG}"
echo "  extra  : '${EXTRA}'"

ssh star2 "sbatch" << EOF
#!/bin/bash
#SBATCH -J ${NAME}
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --nodelist=${NODE}
#SBATCH --gres=gpu:${GRES}:${NGPU}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${WALL}
#SBATCH --output=${PROJECT}/log/${NAME}-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@300
set -uo pipefail

ATTEMPTS=${PROJECT}/.texgen_v2_alpha_attempts
MAX_ATTEMPTS=3

# Progress is published into the SLURM job Comment. /localscratch is per-node and
# unreadable from the head, and no shared path is reliably mounted there, so
# \`scontrol show job <id>\` is the only live channel. NEVER monitor by attaching
# \`srun --jobid --overlap\` to this job: when step launch is unhealthy that srun
# fails and takes the whole job down (it killed job 236871 on 2026-07-29).
mark() { scontrol update jobid=\${SLURM_JOB_ID} comment="\$*" 2>/dev/null; return 0; }

# Requeue ONLY on the wall-clock signal. A crash must not requeue unboundedly: on
# vulcan an unconditional requeue turned one persistent NCCL failure into ~19
# silent retries across three days, all appended into one log.
handle_timeout() {
    echo "[trap] wall-clock signal (~5 min left); stopping for requeue..."
    mark "requeue at wall-clock"
    if [ -n "\${TRAIN_PID:-}" ] && kill -0 "\${TRAIN_PID}" 2>/dev/null; then
        kill -TERM "\${TRAIN_PID}" 2>/dev/null
        wait "\${TRAIN_PID}" || true
    fi
    echo 0 > "\$ATTEMPTS"          # progress was made; reset the crash counter
    scontrol requeue "\${SLURM_JOB_ID}" || true
    exit 0
}
trap handle_timeout SIGTERM

mark "starting"
echo "=== ${NAME}  job \${SLURM_JOB_ID}  node \$(hostname)  \$(date -Iseconds) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /localscratch/dya78/miniconda3/etc/profile.d/conda.sh
conda activate texgen-bw
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}
# torch 2.9 renamed this; set both so it is honoured either way. Not cosmetic —
# the vulcan OOM explicitly recommended expandable_segments.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR=/localscratch/dya78/tmp
export HF_HOME=${PROJECT}/.cache/huggingface
# The CLIP tokenizer/text-encoder come from stabilityai/stable-diffusion-3.5-large,
# which is a GATED repo: without a cache the node gets
#   401 Unauthorized ... /stable-diffusion-3.5-large/resolve/main/tokenizer/vocab.json
#   huggingface_hub.errors.GatedRepoError
# (job 237299 crashed exactly there — it was NOT an OOM). The cache is staged from
# the workstation via jupiter rather than putting an HF token on the cluster, and
# OFFLINE stops any attempt to re-validate against the hub. vulcan's script sets the
# same flag for the same reason.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# flash-attn is not available on sm_120 (two source builds, ~45 min and ~65 min,
# both left it unimportable — jobs 236972/237121, confirmed by 237198). This selects
# the dense attention path instead. It is a recorded DEVIATION from the reference run:
# numerics differ in the safe direction (no .half() cast) and windowing uses
# min(patch_size_max, n_points) rather than a fixed patch size.
export TEXGEN_ENABLE_FLASH=0

# Lightning: without this it detects SLURM and uses its SLURM launcher, which
# requires ntasks == devices. One task with 4 devices then hangs at
# "Initializing distributed" — which looks exactly like a NCCL problem and is not
# one. All three TRELLIS2/script/star2/*.sh set it for the same reason.
export SLURM_JOB_NAME=bash

mkdir -p \$TMPDIR ${PROJECT}/log \$HF_HOME

mark "checking data"
for f in ${PROJECT}/data/df_SomgProc_final.parquet \\
         ${PROJECT}/data/data_splits_emissive_74k_stratified_newbake_vae.json \\
         ${PROJECT}/data/texgen_train_root/thumbnails; do
    [ -e "\$f" ] || { echo "MISSING: \$f"; mark "FATAL missing \$(basename \$f)"; exit 2; }
done

# Alpha pre-flight. A PARTIALLY delivered alpha root is the failure mode this run is most
# exposed to, and it is completely silent: the loader raises, __getitem__ used to return
# None, and collate_fn drops the sample before it inspects any key -- so the survivors stack
# cleanly and training just continues on a smaller, biased batch. The loader now re-raises
# instead, but a crash 30 hours in is still a wasted allocation. So COUNT the whole tree
# here, do not sample it.
mark "checking alpha"
ALPHA_HAVE=\$(find ${PROJECT}/data/texgen_train_root -name alpha.npy | wc -l)
ALPHA_WANT=\$(find ${PROJECT}/data/texgen_train_root -name somage.npz | wc -l)
echo "[alpha] sidecars: \$ALPHA_HAVE / \$ALPHA_WANT somage.npz"
if [ "\$ALPHA_HAVE" -ne "\$ALPHA_WANT" ]; then
    echo "FATAL: \$(( ALPHA_WANT - ALPHA_HAVE )) shapes have no alpha.npy sidecar"
    mark "FATAL alpha incomplete (\$ALPHA_HAVE/\$ALPHA_WANT)"
    exit 3
fi
python - <<'ALPHACHK' || { mark "FATAL bad alpha sidecar"; exit 3; }
import glob, os, sys, random
import numpy as np
root = "${PROJECT}/data/texgen_train_root"
ps = glob.glob(os.path.join(root, "*", "*", "somage.npz"))
if not ps:
    sys.exit("no somage.npz under " + root)
random.seed(0)
for p in random.sample(ps, min(200, len(ps))):
    with np.load(p) as z:
        if "alpha" in z.files:
            continue
    side = os.path.join(os.path.dirname(p), "alpha.npy")
    a = np.load(side)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 1:
        sys.exit("BAD ALPHA SIDECAR %s: shape=%s dtype=%s" % (side, a.shape, a.dtype))
print("[alpha] ok — 200 random sidecars are uint8 (H, W, 1)")
ALPHACHK

cd ${PROJECT}/TEXGen
git pull --ff-only origin ${BRANCH} || true
echo "TEXGen HEAD: \$(git log --oneline -1)"

# wandb has no credentials on a star2 compute node — home is per-node, so there is no
# ~/.netrc and no WANDB_API_KEY. Without this guard the run dies at startup with
#   wandb.errors.errors.UsageError: No API key configured.
# (job 237309, crash 2/3). An API key is a credential and does not go on the cluster;
# AGENTS.md's convention for no-auth nodes is an offline run synced afterwards:
#   wandb sync -p lightgen <offline-run-dir>
if [ -z "\${WANDB_API_KEY:-}" ] && [ ! -f "\$HOME/.netrc" ]; then
    export WANDB_MODE=offline
    echo "[wandb] no credentials on this node — OFFLINE; sync later with:"
    echo "        wandb sync -p lightgen ${PROJECT}/TEXGen/wandb/offline-run-*"
fi

mark "training"
# No \`| tee\`: SLURM's --output already persists everything, and piping would make
# TRAIN_PID the tee process, so \`wait\` would report TEE's exit status — a crashed
# trainer would be read as success. Backgrounded bare so the SIGTERM trap can reach
# it and \$? is python's own status.
python launch.py --config ${CONFIG} --gpu ${GPUS} --train --wandb ${EXTRA} &
TRAIN_PID=\$!
wait "\$TRAIN_PID"
EXIT_CODE=\$?

echo "[done] train exit \${EXIT_CODE} at \$(date -Iseconds)"
if [ "\${EXIT_CODE}" -eq 0 ]; then
    mark "finished ok"
else
    N=\$(( \$(cat "\$ATTEMPTS" 2>/dev/null || echo 0) + 1 ))
    echo "\$N" > "\$ATTEMPTS"
    if [ "\$N" -lt "\$MAX_ATTEMPTS" ]; then
        echo "[done] crash \${N}/\${MAX_ATTEMPTS} — requeueing"
        mark "crash \${N}/\${MAX_ATTEMPTS}, requeued"
        scontrol requeue "\${SLURM_JOB_ID}" || true
    else
        echo "[done] crash \${N}/\${MAX_ATTEMPTS} — NOT requeueing; inspect the log"
        mark "crash \${N}/\${MAX_ATTEMPTS}, HELD - inspect log"
    fi
fi
exit \$EXIT_CODE
EOF

echo "Submitted. Watch with:"
echo "  ssh star2 'squeue -u dya78 -o \"%.10i %.24j %.10T %.10M %R\"'"
echo "  ssh star2 'scontrol show job <ID> | tr \" \" \"\\n\" | grep ^Comment='"
