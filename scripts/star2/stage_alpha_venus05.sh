#!/usr/bin/env bash
# Stage the alpha sidecars onto cs-venus-05's node-local /localscratch.
#
# Run FROM THE WORKSTATION. star2's /localscratch is per-node and unreachable from the
# head, and cs-venus-05 cannot see the jupiter NFS tree at all -- but it CAN ssh back to
# the workstation, so the node pulls the pack itself from inside its allocation.
#
# The pack is ~0.56 GB holding 72,421 `<shard>/<sha>/alpha.npy` members, built by
#   python -m data_processing.uv_voxel_pipeline.export_alpha_pack
# Extracting it beside each existing somage.npz upgrades the staged 12ch training root
# in place WITHOUT touching the 383 GB of npz -- so the no-alpha ablation row keeps
# training on byte-identical data and stays reproducible.
#
# Idempotent: re-running overwrites the sidecars with the same bytes.
#
# Usage: bash scripts/star2/stage_alpha_venus05.sh [--dry-run]
set -euo pipefail

NODE=cs-venus-05
PROJECT=/localscratch/dya78/lightgen
SRC_HOST=${SRC_HOST:-dya78@cs-3dlg-23.cmpt.sfu.ca}
SRC_PACK=${SRC_PACK:-/local-scratch/localhome/dya78/lightgen_74k_staging/alpha_pack.tar.gz}
CPUS=${CPUS:-16}          # a shared node: take a modest slice, not all 128 cores
MEM=${MEM:-32G}
WALL=${WALL:-01:00:00}
EXPECT=${EXPECT:-72421}

echo "[stage-alpha] node=${NODE} pack=${SRC_PACK} expect=${EXPECT}"

# NOTE: no -p/--partition. On star2, -w + --time is the whole interface; naming a
# partition adds no capability and several ways to be wrong.
ssh star2 "sbatch" << EOF
#!/usr/bin/env bash
#SBATCH --job-name=stage_alpha
#SBATCH --nodes=1
#SBATCH -w ${NODE}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${WALL}
#SBATCH --output=${PROJECT}/log/stage_alpha_%j.log

set -euo pipefail
ROOT=${PROJECT}/data/texgen_train_root
PACK=${PROJECT}/data/alpha_pack.tar.gz

echo "[stage-alpha] host=\$(hostname) start=\$(date -Is)"
df -h /localscratch | tail -1

echo "[stage-alpha] pulling the pack from the workstation"
rsync -ah --info=progress2 ${SRC_HOST}:${SRC_PACK} \$PACK
ls -la \$PACK

echo "[stage-alpha] extracting into \$ROOT"
tar -xzf \$PACK -C \$ROOT
echo "[stage-alpha] extraction done \$(date -Is)"

HAVE=\$(find \$ROOT -name alpha.npy | wc -l)
WANT=\$(find \$ROOT -name somage.npz | wc -l)
echo "[stage-alpha] alpha.npy=\$HAVE somage.npz=\$WANT expect=${EXPECT}"
if [ "\$HAVE" -ne "${EXPECT}" ] || [ "\$HAVE" -ne "\$WANT" ]; then
    echo "[stage-alpha] FATAL: sidecar count mismatch"
    exit 2
fi

# every sidecar must be the uint8 (H, W, 1) plane the loader's /255 decode assumes;
# a float32 plane would pass every existence check and then be silently divided again.
source /localscratch/dya78/miniconda3/etc/profile.d/conda.sh
conda activate texgen-bw
python - <<'PYCHK'
import glob, os, random, sys
import numpy as np
root = "${PROJECT}/data/texgen_train_root"
ps = glob.glob(os.path.join(root, "*", "*", "alpha.npy"))
print("[stage-alpha] sampling %d of %d sidecars" % (min(500, len(ps)), len(ps)))
random.seed(0)
for p in random.sample(ps, min(500, len(ps))):
    a = np.load(p)
    if a.dtype != np.uint8 or a.shape != (512, 512, 1):
        sys.exit("BAD %s shape=%s dtype=%s" % (p, a.shape, a.dtype))
    # a sidecar must sit beside the npz it belongs to
    if not os.path.exists(os.path.join(os.path.dirname(p), "somage.npz")):
        sys.exit("ORPHAN sidecar (no somage.npz beside it): %s" % p)
print("[stage-alpha] sampled sidecars OK")
PYCHK

df -h /localscratch | tail -1
echo "[stage-alpha] DONE \$(date -Is)"
EOF
