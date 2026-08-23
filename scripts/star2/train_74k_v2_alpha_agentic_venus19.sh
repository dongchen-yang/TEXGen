#!/bin/bash
# TEXGen alpha _v2 AGENTIC on cs-venus-19 — the alpha run trained on the AGENTIC-FILTERING
# training set (the QC workflow's rejects removed from train on top of heuristic filtering,
# 63,194 -> 36,255; val 387 / test 388 unchanged).
# 1 node x 4 RTX PRO 6000 Blackwell SE, per-GPU batch 32 -> GLOBAL BATCH 128.
#
# A copy of train_74k_v2_alpha_nonzero_nocopy_venus05.sh with the usual THREE substitutions
# — NAME, CONFIG, and the split filename in the data pre-flight (plus its own ATTEMPTS
# counter, so a crash here does not consume the sibling runs' retry budgets) — and THREE
# additions this sibling did not have:
#   * NODE/GRES/CPUS/MEM/CUDA_VER defaults for cs-venus-19 rather than cs-venus-05;
#   * md5 GATES on the parquet and the split. The split's indices are POSITIONAL into the
#     parquet, so a different parquet silently trains on different shapes, and a rewritten
#     split silently changes the training set — neither is visible in the loss curve. The
#     venus-05 siblings only checked existence; this run gates on content.
#   * ENV GATES on conda, CUDA_HOME and the interpreter. The siblings could assume their
#     node was the one they were built on; this launcher targets a node the env was
#     RSYNCED to, and the run's whole claim is that only the training set differs from
#     the venus-05 arms. `set -uo pipefail` has no -e, so a failed `conda activate` would
#     otherwise sail past and train under whatever python was on PATH.
# Everything else — the SIGTERM/requeue trap, the crash ceiling, SLURM_JOB_NAME=bash,
# TEXGEN_ENABLE_FLASH=0, HF_HUB_OFFLINE, the alpha sidecar count, the wandb-offline
# guard, GPUS=0,1,2,3 — is unchanged, because every one of those lines encodes a failure
# this run would otherwise repeat.
#
# The `#SBATCH --partition=3dlg-hcvc-lab-long` line is inherited verbatim. It is the one
# launcher family in this repo that names a partition, which is against the star2 rule
# (`-w <node>` + `--gres` + `--time` is the whole interface); changing it is out of scope
# here — noted, not fixed, so this run stays comparable to its siblings.
#
# RUNS FROM THE WORKSTATION. It pipes the job body into `ssh star2 sbatch`; the
# script itself never needs to exist on the compute node. That is not a style
# choice: on star2 BOTH /localscratch and /home are per-node, so a path like
# /localscratch/dya78/lightgen does not exist on the head node and
# `ssh star2 'cd <that> && sbatch ...'` cannot work. Same pattern as
# TRELLIS2/script/star2/emission_dit_1k_probe_venus08.sh.
#
# Spec: docs/superpowers/specs/2026-08-22-agentic-filtered-baselines-retrain-design.md
# (parent repo). Plan: docs/superpowers/plans/2026-08-23-agentic-filtered-baselines-retrain.md
# Task 10. Split md5 159492b2b8d104ab63f3d13eeea394d0; SCORE AT epoch 124 / step 35,500.
#
# Usage:
#   bash scripts/star2/train_74k_v2_alpha_agentic_venus19.sh                # full run
#   EXTRA="trainer.max_steps=30 trainer.limit_val_batches=2 checkpoint.dirpath=/localscratch/dya78/lightgen/TEXGen/output_emissive_74k_alpha_v2_agentic_probe/" \
#     bash scripts/star2/train_74k_v2_alpha_agentic_venus19.sh              # datamodule probe
#     # NOTE the dirpath override: a probe must never leave a checkpoint where the real
#     # run's auto_resume would glob it.
#   NODE=cs-venus-05 GRES=rtx_pro_6000_blackwell_max-q CPUS=48 MEM=400G \
#     bash scripts/star2/train_74k_v2_alpha_agentic_venus19.sh              # fallback node
set -euo pipefail

NAME=texgen_alpha_74k_v2_agentic
CONFIG=${CONFIG:-configs/lightgen_pointuv_256_batch32_emissive_74k_v2_alpha_venus19_agentic.yaml}
EXTRA=${EXTRA:-}
# Which local GPUs Lightning sees. VRAM is a per-GPU quantity, so a 1-GPU probe
# answers the sizing question and schedules sooner than a 4-GPU reservation.
GPUS=${GPUS:-0,1,2,3}
NODE=${NODE:-cs-venus-19}
GRES=${GRES:-rtx_pro_6000_blackwell_se}
NGPU=${NGPU:-4}
# venus-19 has 64 cores; take half (shared-node courtesy — the star2 rule is a third to a
# half of a shared box), not venus-05's 48. num_workers 4 x 4 ranks = 16 loader processes
# plus 4 mains, so 32 is comfortably above what the job actually spawns.
CPUS=${CPUS:-32}
# 200G was not enough on venus-05: job 237314 died OUT_OF_MEMORY with MaxRSS 209,710,500K,
# i.e. exactly the 200G ceiling. Each of the 20 processes (4 ranks x 1 main + 4 workers)
# holds a copy of the 824,858-row sample index built from the parquet, and wandb online
# adds upload buffers on top. venus-19 has 755 GB (RealMemory=773164M, measured via
# `scontrol show node` 2026-08-23) and is shared, so take 256G — 28% above
# the measured failure point, on a config with 4 workers rather than the 6 that produced it.
# If this OOMs, raise to 400G rather than cutting num_workers (that changes throughput).
MEM=${MEM:-256G}
WALL=${WALL:-7-00:00:00}
# Verified per-node: Task 9's staging step records `ls -d /usr/local/cuda-*` on venus-19.
# Overridable so a node with a different toolkit needs no edit to this file.
CUDA_VER=${CUDA_VER:-12.9}
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

ATTEMPTS=${PROJECT}/.texgen_v2_alpha_agentic_attempts
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

CONDA_SH=/localscratch/dya78/miniconda3/etc/profile.d/conda.sh
[ -f "\$CONDA_SH" ] || { echo "FATAL: no conda at \$CONDA_SH — bootstrap this node first"; mark "FATAL no conda"; exit 4; }
source "\$CONDA_SH"
conda activate texgen-bw || { echo "FATAL: conda activate texgen-bw failed"; mark "FATAL no texgen-bw env"; exit 4; }
export CUDA_HOME=/usr/local/cuda-${CUDA_VER}
if [ ! -d "\$CUDA_HOME" ]; then
    echo "FATAL: \$CUDA_HOME does not exist on \$(hostname); this node has:"
    ls -d /usr/local/cuda-* 2>/dev/null || echo "  (no /usr/local/cuda-*)"
    echo "Re-run with CUDA_VER=<ver> to match."
    mark "FATAL no \$CUDA_HOME"
    exit 4
fi
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

# Assert the interpreter is the rsynced texgen-bw and that every native extension this
# run needs actually imports. Same smoke as the bootstrap's, run on EVERY start (including
# requeues) because a node can be re-staged between attempts.
mark "checking env"
python - <<'ENVCHK' || { mark "FATAL bad env"; exit 4; }
# Deliberately NOT \`assert\`: an inherited PYTHONOPTIMIZE strips asserts, and this gate
# would then pass silently on the wrong interpreter -- the exact failure it exists to
# catch. sys.exit() cannot be optimised away.
import sys
if "envs/texgen-bw" not in sys.executable:
    sys.exit(f"WRONG INTERPRETER: {sys.executable} (want .../envs/texgen-bw/bin/python)")
import torch, spconv.pytorch, torchsparse, nvdiffrast.torch, pytorch_lightning
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False -- no usable GPU in this allocation")
print(f"[env] {sys.executable}")
print(f"[env] torch {torch.__version__}  cuda {torch.version.cuda}  gpus {torch.cuda.device_count()}"
      f"  lightning {pytorch_lightning.__version__}")
ENVCHK

mark "checking data"
for f in ${PROJECT}/data/df_SomgProc_final.parquet \\
         ${PROJECT}/data/data_splits_emissive_74k_stratified_newbake_vae_agentic.json \\
         ${PROJECT}/data/texgen_train_root/thumbnails; do
    [ -e "\$f" ] || { echo "MISSING: \$f"; mark "FATAL missing \$(basename \$f)"; exit 2; }
done

# CONTENT gates, not existence gates. The split's indices are POSITIONAL into the parquet
# filtered to success==True, so a parquet that is not exactly this one selects DIFFERENT
# SHAPES with no error, and a rewritten split silently changes the training set. Neither
# shows up in the loss curve; both invalidate the comparison this run exists to make.
mark "checking md5"
check_md5() {
    local want=\$1 f=\$2 got
    got=\$(md5sum "\$f" | cut -d' ' -f1)
    if [ "\$got" != "\$want" ]; then
        echo "FATAL md5 mismatch \$f"
        echo "  got  \$got"
        echo "  want \$want"
        mark "FATAL md5 \$(basename \$f)"
        exit 2
    fi
    echo "[md5] ok  \$(basename \$f)  \$got"
}
check_md5 c4196a9dba89e354f47c415ea0167e2c ${PROJECT}/data/df_SomgProc_final.parquet
check_md5 159492b2b8d104ab63f3d13eeea394d0 ${PROJECT}/data/data_splits_emissive_74k_stratified_newbake_vae_agentic.json

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
