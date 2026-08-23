#!/bin/bash
# TEXGen alpha _v2 AGENTIC on **fir** (Alliance Canada) — 1 node x 4 H100 80 GB,
# per-GPU batch 32 -> GLOBAL BATCH 128, identical to the cs-venus-19 run.
#
# WHY THIS FILE EXISTS: cs-venus-19 is held by another user's job until ~2026-08-27 18:00.
# This launcher starts the same recipe on fir so the answer is not gated on that node. The
# venus-19 run remains the parity-clean number (see the config header for what this port
# costs); this is the early, same-recipe-different-site result.
#
# Ported from scripts/star2/train_74k_v2_alpha_agentic_venus19.sh onto the fir sbatch
# template of TRELLIS.2-lightning script/fir/emission_vae_pbr2emission_74k_v3_agentic.sh.
# Deltas from the venus-19 launcher, all forced by the site:
#   * module load + venv instead of conda (fir forbids conda);
#   * $SLURM_TMPDIR staging: fir's INODE quota is 789K/1000K used (~211K free) and the
#     extracted training root is ~290K inodes, so it CANNOT live on /scratch. The 410 GB
#     stays as 9 tars (9 inodes) and every job segment extracts into $SLURM_TMPDIR
#     (~7.2 TB node-local, no quota). TMPDIR is WIPED at job end -> checkpoints go to
#     /scratch, and a requeue re-extracts (~15 min, paid once per segment);
#   * 24 h segments with --requeue + auto_resume, the fir house style;
#   * a sick-node GPU guard (an orphaned PID held 69 GB on fc10218 on 2026-08-20);
#   * a PIN OVERLAY on PYTHONPATH. bootstrap_texgen_bw.sh deliberately HARD-PINS four
#     packages on venus -- transformers==4.28.1, diffusers==0.28.0,
#     huggingface_hub==0.25.2, omegaconf==2.3.0 -- because the CLIP/diffusers
#     conditioning stack is version-sensitive. fir's shared venv carries transformers
#     4.57.3 / diffusers 0.32.2 / hub 0.36.0, i.e. 29 minor versions of drift on the
#     encoder this model conditions on. Rather than mutate a SHARED (and already
#     dist-info-corrupted) venv, the pins are installed into an isolated --target dir
#     and shadowed in via PYTHONPATH, which sys.path puts ahead of site-packages.
#     Verified on fir: the three versions resolve and the CLIP image encoder loads
#     offline from fir's own HF cache. omegaconf already matches at 2.3.0.
#     Build/rebuild with:
#       pip install --target /scratch/dya78/lightgen/texgen_pin_overlay --no-deps \
#         transformers==4.28.1 diffusers==0.28.0 huggingface_hub==0.25.2 tokenizers==0.13.3
#
# WHAT IS DELIBERATELY UNCHANGED FROM THE VENUS LAUNCHER, because each encodes a failure:
#   * TEXGEN_ENABLE_FLASH=0 -- LOAD-BEARING AND DIFFERENT IN KIND HERE. On the venus nodes
#     flash-attn is unimportable (sm_120), so the flag only records reality. On fir
#     flash_attn 2.8.3 IMPORTS, so without this flag the run would silently take a
#     different attention kernel and different windowing: a second changed variable.
#   * SLURM_JOB_NAME=bash -- stops Lightning using its SLURM launcher (which demands
#     ntasks == devices). One task with 4 devices + native spawn is what this code is
#     validated with; the sibling TRELLIS.2 launcher takes the other route on purpose.
#   * HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE -- the CLIP stack pulls from a GATED repo and
#     a cache miss is a 401, not a download (that crashed job 237299). fir's own
#     ~/.cache/huggingface holds both models, so HF_HOME is deliberately NOT set.
#   * the alpha sidecar COUNT (not a sample): a partially delivered alpha root is silent.
#   * md5 CONTENT gates on the parquet and split: the split's indices are POSITIONAL into
#     the parquet, so a wrong parquet trains on different shapes with no error.
#
# Usage (runs from the workstation OR from fir; it submits either way):
#   # probe first — separate output dir, so auto_resume can never find it later:
#   OUT_SUFFIX=_probe WALLTIME=3:00:00 \
#     EXTRA="trainer.max_steps=30 trainer.limit_val_batches=2" \
#     bash TEXGen/scripts/fir/train_74k_v2_alpha_agentic_fir.sh
#   # full run:
#   bash TEXGen/scripts/fir/train_74k_v2_alpha_agentic_fir.sh
#
# Spec: lightgen docs/superpowers/specs/2026-08-22-agentic-filtered-baselines-retrain-design.md
# Score at epoch 124 / step 35,500. Split md5 159492b2b8d104ab63f3d13eeea394d0.
set -euo pipefail

NAME=texgen_alpha_74k_v2_agentic
TAG=fir
CONFIG=${CONFIG:-configs/lightgen_pointuv_256_batch32_emissive_74k_v2_alpha_fir_agentic.yaml}
REPO=${REPO:-/scratch/dya78/lightgen/TEXGen_agentic}
DATA=${DATA:-/scratch/dya78/lightgen/data}
RUNS=${RUNS:-/scratch/dya78/lightgen/texgen_runs}
OVERLAY=${OVERLAY:-/scratch/dya78/lightgen/texgen_pin_overlay}
BRANCH=texgen-74k-v2-venus05
OUT_SUFFIX=${OUT_SUFFIX:-}
EXTRA=${EXTRA:-}
NUM_GPUS=${NUM_GPUS:-4}
BS=${BS:-32}
GLOBAL_BATCH=${GLOBAL_BATCH:-128}
CPUS=${CPUS:-48}
MEM=${MEM:-1024G}
WALLTIME=${WALLTIME:-24:00:00}
ACCOUNT=${ACCOUNT:-rrg-msavva_gpu}

SPLIT=data_splits_emissive_74k_stratified_newbake_vae_agentic.json
SHAS=agentic_train_shas.txt
PARQUET_MD5=c4196a9dba89e354f47c415ea0167e2c
SPLIT_MD5=159492b2b8d104ab63f3d13eeea394d0
SHAS_MD5=191a91bc295192517565eaac6089753b
SUCCESS_ROWS=824858          # rows of the parquet with success==True (measured, pandas 2.3.3)
EXPECT=72421
OUTPUT_DIR=${RUNS}/output_emissive_74k_alpha_v2_agentic_fir${OUT_SUFFIX}

# The global batch is the thing being held equal across sites and arms. Refuse rather than
# quietly train a different effective batch than the venus-19 run.
ACTUAL=$((NUM_GPUS * BS))
if [ "${ACTUAL}" -ne "${GLOBAL_BATCH}" ]; then
    echo "REFUSING TO SUBMIT: NUM_GPUS(${NUM_GPUS}) x BS(${BS}) = ${ACTUAL}, expected ${GLOBAL_BATCH}." >&2
    exit 1
fi

echo "Submitting ${NAME}${OUT_SUFFIX}_${TAG}"
echo "  gpus     : ${NUM_GPUS} x h100  (BS ${BS} -> global ${ACTUAL})"
echo "  config   : ${CONFIG}"
echo "  outdir   : ${OUTPUT_DIR}"
echo "  walltime : ${WALLTIME}   account: ${ACCOUNT}"
echo "  extra    : '${EXTRA}'"

if command -v sbatch >/dev/null 2>&1; then SUBMIT=(sbatch); else SUBMIT=(ssh -o BatchMode=yes fir sbatch); fi

"${SUBMIT[@]}" << EOF
#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}${OUT_SUFFIX}_${TAG}
#SBATCH --gpus-per-node=h100:${NUM_GPUS}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${WALLTIME}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=%N-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@900

set -uo pipefail

handle_timeout() {
    echo "[trap] wall-clock signal (~15 min left); stopping for requeue..."
    if [ -n "\${TRAIN_PID:-}" ] && kill -0 "\${TRAIN_PID}" 2>/dev/null; then
        kill -TERM "\${TRAIN_PID}" 2>/dev/null
        wait "\${TRAIN_PID}" || true
    fi
    scontrol requeue "\${SLURM_JOB_ID}" || true
    exit 0
}
trap handle_timeout SIGTERM

echo "======================================"
echo "Job: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})  restart_count=\${SLURM_RESTART_COUNT:-0}"
echo "Started: \$(date -Iseconds)  Node: \$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "======================================"

# fc10218 was found with an orphaned PID holding 69 GB on GPU 0 (2026-08-20). Refuse to
# start on a card that is already occupied — at per-GPU batch 32 this run needs ~69 GB of
# the 80 GB card and would OOM in a way that looks like a config problem.
BUSY=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '\$1 > 2048' | wc -l)
if [ "\${BUSY}" -gt 0 ]; then
    echo "FATAL: \${BUSY} GPU(s) on \$(hostname) already hold >2 GB before training; node is sick. Requeueing."
    scontrol requeue "\${SLURM_JOB_ID}"; sleep 60; exit 0
fi

module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash

cd ${REPO}
git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}
echo "TEXGen HEAD: \$(git log --oneline -1)"

VENV=/scratch/dya78/lightgen/env
[ -f "\${VENV}/bin/activate" ] || { echo "FATAL: no venv at \${VENV}"; exit 4; }
source "\${VENV}/bin/activate"

# ---- environment gates -------------------------------------------------------------
# fir has no conda, so the venus launcher's 'envs/texgen-bw' interpreter assertion is
# replaced by this venv's own prefix. Deliberately NOT \`assert\` — an inherited
# PYTHONOPTIMIZE strips asserts and the gate would pass silently on the wrong python.
export TEXGEN_ENABLE_FLASH=0
export SLURM_JOB_NAME=bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export TMPDIR=\${SLURM_TMPDIR}/tmp
mkdir -p "\$TMPDIR"

# Shadow the shared venv's newer conditioning stack with venus's hard pins. PYTHONPATH
# precedes site-packages in sys.path, so these win without touching the shared venv.
[ -d "${OVERLAY}/transformers" ] || { echo "FATAL: pin overlay missing at ${OVERLAY} (see the header for the pip line)"; exit 4; }
export PYTHONPATH=${OVERLAY}\${PYTHONPATH:+:\$PYTHONPATH}

python - <<'ENVCHK' || { echo "FATAL bad env"; exit 4; }
import sys, os
if "/scratch/dya78/lightgen/env" not in sys.executable:
    sys.exit("WRONG INTERPRETER: %s" % sys.executable)
if os.environ.get("TEXGEN_ENABLE_FLASH") != "0":
    sys.exit("TEXGEN_ENABLE_FLASH must be 0 on fir: flash_attn imports here but not on the venus arms")
import transformers, diffusers, huggingface_hub, omegaconf
# The four packages bootstrap_texgen_bw.sh hard-pins on venus. Anything else in this env
# is allowed to differ (this is an acknowledged cross-site run), but the conditioning
# stack the model actually consumes must match, or the comparison loses its meaning.
for name, mod, want in (("transformers", transformers, "4.28.1"),
                        ("diffusers", diffusers, "0.28.0"),
                        ("huggingface_hub", huggingface_hub, "0.25.2"),
                        ("omegaconf", omegaconf, "2.3.0")):
    got = mod.__version__.split("+")[0]
    if got != want:
        sys.exit("%s is %s but venus pins %s -- pin overlay not shadowing (PYTHONPATH=%s)"
                 % (name, got, want, os.environ.get("PYTHONPATH", "<unset>")))
import torch, spconv.pytorch, torchsparse, nvdiffrast.torch, pytorch_lightning
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False -- no usable GPU in this allocation")
print("[env] pins ok: transformers %s / diffusers %s / hub %s / omegaconf %s"
      % (transformers.__version__, diffusers.__version__, huggingface_hub.__version__, omegaconf.__version__))
print("[env] %s" % sys.executable)
print("[env] torch %s  cuda %s  gpus %d  lightning %s"
      % (torch.__version__, torch.version.cuda, torch.cuda.device_count(), pytorch_lightning.__version__))
ENVCHK

# ---- content gates on the source data (cheap, on /scratch, before the 410 GB extract) --
check_md5() {
    local want=\$1 f=\$2 got
    [ -f "\$f" ] || { echo "FATAL missing \$f"; exit 2; }
    got=\$(md5sum "\$f" | cut -d' ' -f1)
    if [ "\$got" != "\$want" ]; then
        echo "FATAL md5 mismatch \$f"; echo "  got  \$got"; echo "  want \$want"; exit 2
    fi
    echo "[md5] ok  \$(basename \$f)  \$got"
}
check_md5 ${PARQUET_MD5} ${DATA}/df_SomgProc_final.parquet
check_md5 ${SPLIT_MD5}   ${DATA}/${SPLIT}
check_md5 ${SHAS_MD5}    ${DATA}/${SHAS}

for t in ${DATA}/texgen_tars_v2/texgen_root_chunk_0{0,1,2,3,4,5,6,7}.tar \\
         ${DATA}/texgen_tars_v2/thumbnails_emissive.tar \\
         ${DATA}/alpha_pack.tar.gz; do
    [ -s "\$t" ] || { echo "FATAL: missing or empty \$t — staging incomplete"; exit 2; }
done
echo "[data] all 10 archives present on /scratch"

# ---- stage into \$SLURM_TMPDIR ---------------------------------------------------------
D=\${SLURM_TMPDIR}/lightgen/data
ROOT=\$D/texgen_train_root
mkdir -p "\$ROOT"

AVAIL_KB=\$(df -Pk "\${SLURM_TMPDIR}" | awk 'NR==2{print \$4}')
echo "[stage] \\\$SLURM_TMPDIR=\${SLURM_TMPDIR}  free=\$(( AVAIL_KB / 1024 / 1024 )) GB"
if [ "\$AVAIL_KB" -lt 471859200 ]; then      # 450 GB
    echo "FATAL: \\\$SLURM_TMPDIR has < 450 GB free; the training root needs ~410 GB"
    exit 2
fi

cp ${DATA}/df_SomgProc_final.parquet ${DATA}/${SPLIT} ${DATA}/${SHAS} "\$D/"

echo "[stage] extracting 8 chunk tars in parallel  \$(date -Iseconds)"
printf '%s\n' ${DATA}/texgen_tars_v2/texgen_root_chunk_0{0,1,2,3,4,5,6,7}.tar \\
  | xargs -P 4 -I{} tar xf {} -C "\$ROOT" || { echo "FATAL: chunk extraction failed"; exit 2; }
echo "[stage] extracting thumbnails + alpha sidecars  \$(date -Iseconds)"
tar xf  ${DATA}/texgen_tars_v2/thumbnails_emissive.tar -C "\$ROOT" || { echo "FATAL: thumbnails extract failed"; exit 2; }
tar xzf ${DATA}/alpha_pack.tar.gz                      -C "\$ROOT" || { echo "FATAL: alpha extract failed"; exit 2; }
echo "[stage] extraction done  \$(date -Iseconds)"

# COUNT the whole tree, do not sample it: a partially delivered root is completely silent
# (the loader used to return None and collate_fn dropped the sample, so training just
# continued on a smaller, biased batch).
NPZ=\$(find "\$ROOT" -name somage.npz | wc -l)
ALP=\$(find "\$ROOT" -name alpha.npy  | wc -l)
THM=\$(ls "\$ROOT/thumbnails" 2>/dev/null | wc -l)
echo "[stage] somage.npz=\$NPZ  alpha.npy=\$ALP  thumbnails=\$THM  (expect ${EXPECT} each)"
if [ "\$NPZ" -ne "${EXPECT}" ] || [ "\$ALP" -ne "${EXPECT}" ] || [ "\$THM" -lt "${EXPECT}" ]; then
    echo "FATAL: staged tree is incomplete"
    exit 2
fi

python - <<'ALPHACHK' || { echo "FATAL bad alpha sidecar"; exit 3; }
import glob, os, random, sys
import numpy as np
root = os.path.join(os.environ["SLURM_TMPDIR"], "lightgen", "data", "texgen_train_root")
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
print("[alpha] ok - 200 random sidecars are uint8 (H, W, 1)")
ALPHACHK

# ---- the split -> parquet POSITIONAL MAPPING gate --------------------------------------
# The split file stores POSITIONAL indices into the parquet filtered to success==True, and
# the loader resolves them as samples[i] over df.iterrows() in row order. So the identity of
# the 36,255 training shapes depends on pandas' read_parquet + boolean-filter row ORDER, not
# just on the two md5s above. fir runs pandas 3.0.x where the venus arms ran a 2.2/2.3-era
# pandas, and ANY drift there would train this run on different shapes with no error and no
# visible symptom. So reproduce the mapping and require it to equal agentic_train_shas.txt,
# which was derived independently from the same two files (verified byte-exact on the
# workstation under pandas 2.3.3: 824,858 success rows, order-exact match).
export SPLIT_NAME=${SPLIT}
export SHAS_NAME=${SHAS}
export WANT_SUCCESS_ROWS=${SUCCESS_ROWS}
python - <<'SPLITCHK' || { echo "FATAL split/parquet positional mapping"; exit 2; }
import json, os, sys
import pandas as pd
d = os.path.join(os.environ["SLURM_TMPDIR"], "lightgen", "data")
df = pd.read_parquet(os.path.join(d, "df_SomgProc_final.parquet"))
if "success" in df.columns:
    df = df[df["success"] == True]
n = len(df)
want_n = int(os.environ["WANT_SUCCESS_ROWS"])
if n != want_n:
    sys.exit("success==True rows = %d, expected %d (pandas %s changed read/filter semantics)"
             % (n, want_n, pd.__version__))
ids = [str(x) for x in df.index]
sp = json.load(open(os.path.join(d, os.environ["SPLIT_NAME"])))
tr = sp["train"]["indices"]; va = sp["val"]["indices"]; te = sp["test"]["indices"]
if (len(tr), len(va), len(te)) != (36255, 387, 388):
    sys.exit("split counts (%d, %d, %d), expected (36255, 387, 388)" % (len(tr), len(va), len(te)))
hi = max(max(tr), max(va), max(te))
if hi >= n:
    sys.exit("split index %d is out of range for %d rows" % (hi, n))
got = [ids[i] for i in tr]
want = [l.strip() for l in open(os.path.join(d, os.environ["SHAS_NAME"])) if l.strip()]
if got != want:
    same_set = sorted(got) == sorted(want)
    sys.exit("TRAIN SHAS DIFFER from %s (same set: %s) under pandas %s -- the positional "
             "indices no longer select the intended shapes"
             % (os.environ["SHAS_NAME"], same_set, pd.__version__))
print("[split] ok - pandas %s: %d success rows; the 36,255 train indices reproduce %s exactly"
      % (pd.__version__, n, os.environ["SHAS_NAME"]))
SPLITCHK

mkdir -p ${OUTPUT_DIR} ${RUNS}
echo "[inodes] /scratch usage before training:"; diskusage_report 2>/dev/null | grep scratch || true

# No \`| tee\`: --output already persists everything, and piping would make TRAIN_PID the
# tee process, so \`wait\` would report TEE's status and a crashed trainer would read as
# success. Backgrounded bare so the SIGTERM trap can reach it.
python launch.py --config ${CONFIG} --gpu \$(seq -s, 0 \$(( ${NUM_GPUS} - 1 ))) --train --wandb \\
    checkpoint.dirpath=${OUTPUT_DIR}/ ${EXTRA} &
TRAIN_PID=\$!
wait "\$TRAIN_PID"
EXIT_CODE=\$?

echo "[done] train exit \${EXIT_CODE} at \$(date -Iseconds)"
if [ "\${EXIT_CODE}" -ne 0 ]; then
    if [ "\${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then
        echo "[done] crash — requeueing (restart_count=\${SLURM_RESTART_COUNT:-0})"
        scontrol requeue "\${SLURM_JOB_ID}" || true
        exit 0
    fi
    echo "[done] crash — restart cap reached, NOT requeueing; inspect the log"
fi
exit \$EXIT_CODE
EOF

echo "Submitted: ${NAME}${OUT_SUFFIX}_${TAG}"
echo "Watch with:"
echo "  ssh -o BatchMode=yes fir 'squeue -u dya78 -o \"%.10i %.34j %.2t %.11M %.11L %R\"'"
echo "  ssh -o BatchMode=yes fir 'tail -40 ~/*-<JOBID>.out'"
