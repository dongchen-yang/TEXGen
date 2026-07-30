#!/bin/bash
# Build the `texgen-bw` conda env on a star2 Blackwell node (sm_120) for TEXGen.
#
# Runs INSIDE a SLURM allocation on the target node (cs-venus-05). Idempotent:
# every stage presence-checks before doing work, so re-running after a failure
# resumes rather than rebuilds.
#
# Why a separate env from the workstation's `texgen`: that one is torch 2.1.0 +
# cu11.8, whose arch list tops out at sm_90. RTX PRO 6000 Blackwell is sm_120, so
# torch and every compiled extension have to be rebuilt.
#
# Extensions are verified with a REAL GPU FORWARD, not an import. A wheel built
# for another arch imports fine and only fails at the first kernel launch with
# "no kernel image is available for execution on the device" — which would surface
# mid-training instead of here.
#
# Gate: spconv. texgen_network.py:758 hardcodes use_cpe=True and the CPE block is
# spconv.SubMConv3d (ptv3_model_texgen.py:542). There is no fallback; if spconv
# cannot run on sm_120 this node is not viable and the run moves to single-node
# vulcan. flash-attn DOES have a fallback (see TEXGEN_ENABLE_FLASH), so its
# failure is recorded and the build continues.

D=/localscratch/dya78
ROOT=$D/lightgen
ENV_NAME=texgen-bw
PY_VER=3.11
CUDA_VER=12.9

export TMPDIR=$D/tmp
export CUDA_HOME=/usr/local/cuda-${CUDA_VER}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="12.0"
# Overridable. Each nvcc worker compiling Blackwell kernels can take several GB of
# HOST RAM, and 8 of them killed job 236972 with SLURM OUT_OF_MEMORY at 64 GB after
# nearly two hours. Pair a lower value with a larger --mem rather than trusting
# either alone.
export MAX_JOBS=${MAX_JOBS:-8}
mkdir -p "$TMPDIR" "$ROOT/log"

SPCONV_RESULT=UNKNOWN
FLASH_RESULT=UNKNOWN

# Progress is published into the SLURM job Comment as well as stdout. star2's
# per-node /localscratch is not readable from the head node and no shared path is
# reliably mounted there, so `scontrol show job <id>` is the only channel that
# reports progress live. Do NOT monitor by attaching `srun --jobid --overlap` to a
# running job: when step launch is unhealthy that srun fails and takes the whole
# job down with it (this killed job 236871 on 2026-07-29).
mark() {
    [ -n "${SLURM_JOB_ID:-}" ] && scontrol update jobid=$SLURM_JOB_ID comment="$*" 2>/dev/null
    return 0
}
say() { echo; echo "=============== $* ==============="; STAGE="$*"; mark "$*"; }

# Build output goes to its own file so a failure can be diagnosed from one line
# instead of a chain of probe jobs. The Comment channel does NOT survive the job
# (sacct does not store it), and /localscratch is unreadable from the head node, so
# the reason is also appended to a PERSISTENT file that the next job publishes.
FAILFILE="$ROOT/log/last_failure.txt"
STAGE="init"
STAGE_LOG=""
build_log() { STAGE_LOG="$ROOT/log/build_$1.log"; echo "$STAGE_LOG"; }
first_error() {
    [ -n "${1:-}" ] && [ -f "$1" ] || return 0
    grep -aiE "fatal error|error:|No such file|cannot open|undefined reference|ModuleNotFound" "$1" \
        2>/dev/null | head -1 | cut -c1-240
}

die() {
    local why="$*"
    local detail
    detail=$(first_error "$STAGE_LOG")
    echo
    echo "!!!!!!! FATAL: $why !!!!!!!"
    [ -n "$detail" ] && echo "!!!!!!! first error: $detail"
    mkdir -p "$ROOT/log"
    {
        echo "=== $(date -Iseconds) job ${SLURM_JOB_ID:-?} stage='$STAGE' ==="
        echo "reason: $why"
        [ -n "$detail" ] && echo "first_error: $detail"
        [ -n "$STAGE_LOG" ] && [ -f "$STAGE_LOG" ] && { echo "--- tail of $STAGE_LOG ---"; tail -25 "$STAGE_LOG"; }
    } >> "$FAILFILE"
    mark "FATAL: $why"
    summary
    exit 1
}

summary() {
    echo
    echo "########## BOOTSTRAP SUMMARY ##########"
    echo "spconv     : $SPCONV_RESULT"
    echo "flash-attn : $FLASH_RESULT"
    echo "#######################################"
}

say "[0/11] preserve pre-existing uncommitted work in $ROOT"
# /localscratch is persistent on star2 and we do not delete things there. This
# tree is a Nov/Dec-2025 checkout of the parent repo carrying uncommitted edits to
# data_processing/render_based/* — files that no longer exist upstream. Preserve
# them two independent ways, then leave the tree clean for our use. The parent
# repo is NOT pulled: it is private and this node has no GitHub key.
if [ -d "$ROOT/.git" ]; then
    DIRTY=$(git -C "$ROOT" status --porcelain --ignore-submodules=all | wc -l)
    if [ "$DIRTY" -gt 0 ]; then
        TS=$(date +%Y%m%d-%H%M%S)
        mkdir -p "$D/.superseded"
        PATHS=$(git -C "$ROOT" status --porcelain --ignore-submodules=all | awk '{print $NF}')
        EXISTING=""
        for p in $PATHS; do [ -e "$ROOT/$p" ] && EXISTING="$EXISTING $p"; done
        if [ -n "$EXISTING" ]; then
            tar czf "$D/.superseded/${TS}-lightgen-dec2025-worktree.tgz" -C "$ROOT" $EXISTING \
                && echo "tar  -> $D/.superseded/${TS}-lightgen-dec2025-worktree.tgz"
        fi
        git -C "$ROOT" stash push --include-untracked -m "dec-2025 render_based work, preserved ${TS}" \
            && echo "stash -> $(git -C "$ROOT" stash list | head -1)"
    else
        echo "worktree already clean — nothing to preserve"
    fi
    echo "status after: $(git -C "$ROOT" status --porcelain --ignore-submodules=all | wc -l) entries (expect 0)"
else
    echo "no parent git repo at $ROOT — nothing to preserve"
    mkdir -p "$ROOT"
fi

say "[1/11] miniconda at $D/miniconda3"
if [ ! -d $D/miniconda3 ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $D/miniconda.sh || die "miniconda download"
    bash $D/miniconda.sh -b -p $D/miniconda3 || die "miniconda install"
else
    echo "already present"
fi
source $D/miniconda3/etc/profile.d/conda.sh || die "conda profile"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null

say "[2/11] conda env $ENV_NAME (python $PY_VER)"
conda env list | grep -q "^${ENV_NAME} " || conda create -y -n $ENV_NAME python=$PY_VER || die "conda create"
conda activate $ENV_NAME || die "conda activate"
python -V

say "[3/11] torch cu128 — must carry sm_120 kernels"
python -c "import torch" 2>/dev/null || \
    pip install -q "torch==2.9.*" torchvision --index-url https://download.pytorch.org/whl/cu128 || die "torch install"
python - <<'EOF' || exit 1
import torch, sys
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arch_list", torch.cuda.get_arch_list())
print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
if "sm_120" not in torch.cuda.get_arch_list():
    print("FATAL: torch has no sm_120 kernels — wrong wheel index"); sys.exit(1)
if torch.cuda.get_device_capability(0) != (12, 0):
    print("FATAL: device is not sm_120"); sys.exit(1)
print("TORCH OK")
EOF
[ $? -ne 0 ] && die "torch sm_120 check"

say "[4/11] spconv — THE GATE (real SubMConv3d forward+backward)"
python -c "import spconv.pytorch" 2>/dev/null || pip install -q spconv-cu128 || pip install -q spconv-cu126
if python - <<'EOF'
import torch, spconv.pytorch as spconv
feats  = torch.randn(1024, 32, device='cuda')
coords = torch.randint(0, 32, (1024, 4), dtype=torch.int32, device='cuda'); coords[:, 0] = 0
x    = spconv.SparseConvTensor(feats, coords, [32, 32, 32], 1)
conv = spconv.SubMConv3d(32, 32, kernel_size=3, bias=True, indice_key="probe").cuda()
y = conv(x); y.features.sum().backward()
print("SPCONV OK", tuple(y.features.shape))
EOF
then
    SPCONV_RESULT="PASS ($(python -c 'import spconv; print(getattr(spconv,"__version__","?"))' 2>/dev/null))"
else
    SPCONV_RESULT="FAIL"
    die "spconv cannot run on sm_120 — venus-05 is not viable; move to single-node vulcan"
fi

say "[5/11] flash-attn (has a fallback — failure is recorded, not fatal)"
# SKIP_FLASH=1 goes straight to the fallback. Set it after a build has already been
# tried and failed: on sm_120 the source build ran ~45 min in job 236972 and ~65 min
# in 237121 and left flash_attn unimportable both times (verified independently by
# job 237198). flash-attn 2.x targets Ampere/Ada/Hopper, so Blackwell being
# unsupported is the expected outcome, not a misconfiguration — and there IS a dense
# attention path (ptv3_model_texgen.py:401-418), reached via TEXGEN_ENABLE_FLASH=0.
if [ "${SKIP_FLASH:-0}" = "1" ] && ! python -c "import flash_attn" 2>/dev/null; then
    echo "SKIP_FLASH=1 — not attempting the build; the dense attention fallback will be used"
elif ! python -c "import flash_attn" 2>/dev/null; then
    pip install -q packaging ninja
    echo "building flash-attn from source (expect ~1h)..."
    pip install flash-attn --no-build-isolation
fi
if python - <<'EOF'
import torch, flash_attn
qkv = torch.randn(256, 3, 4, 64, device='cuda', dtype=torch.half)
cu  = torch.tensor([0, 128, 256], dtype=torch.int32, device='cuda')
out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu, max_seqlen=128, dropout_p=0.0, softmax_scale=0.125)
print("FLASH OK", flash_attn.__version__, tuple(out.shape))
EOF
then
    FLASH_RESULT="PASS ($(python -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null))"
else
    FLASH_RESULT="FAIL — TEXGEN_ENABLE_FLASH=0 fallback required (plan Task 3b)"
    echo "flash-attn unavailable; continuing (the dense attention path exists)"
fi

say "[6/11] pure-python layer"
# lpips --no-deps: its dep chain pulls torchvision and silently upgrades torch,
# which breaks every extension built against the pin.
# TEXGen imports pytorch_lightning (launch.py:83), not lightning.
pip install -q \
    pytorch-lightning omegaconf==2.3.0 einops timm \
    transformers==4.28.1 diffusers==0.28.0 huggingface_hub==0.25.2 accelerate \
    jaxtyping typeguard wandb pandas pyarrow \
    imageio matplotlib trimesh Pillow scipy || die "python layer"
# opencv separately and unpinned: the inherited 4.9.0.80 pin has no py3.11 wheel for
# this platform, so pip fell through to a source build and left cv2 unimportable while
# the packages listed before it installed fine (job 237198 measured cv2=0, timm=1).
# texgen_test.py only needs `import cv2`, not a specific version.
python -c "import cv2" 2>/dev/null || pip install -q opencv-python-headless || die "opencv"
pip install -q lpips --no-deps || die "lpips"

say "[7/11] setuptools sanity (broken dist-info kills every source build)"
SETUP_VER=$(python -c "import setuptools; print(setuptools.__version__)" 2>/dev/null)
echo "setuptools: $SETUP_VER"
if [ "$SETUP_VER" = "0.dev0+unknown" ] || [ -z "$SETUP_VER" ]; then
    echo "repairing broken setuptools metadata..."
    SP=$CONDA_PREFIX/lib/python${PY_VER}/site-packages
    rm -rf $SP/setuptools $SP/setuptools-* $SP/pkg_resources $SP/_distutils_hack $SP/distutils-precedence.pth
    pip install -q --ignore-installed "setuptools==69.5.1" || die "setuptools repair"
    pip install -q sympy mpmath networkx filelock fsspec jinja2 typing-extensions
    python -c "import setuptools; print('repaired ->', setuptools.__version__)"
fi

say "[8/11] torch_scatter"
python -c "import torch_scatter" 2>/dev/null || pip install --no-build-isolation torch-scatter > "$(build_log torch_scatter)" 2>&1 || die "torch_scatter build"
python - <<'EOF' || exit 1
import torch, torch_scatter
src = torch.randn(100, 8, device='cuda'); idx = torch.randint(0, 10, (100,), device='cuda')
print("SCATTER OK", tuple(torch_scatter.scatter_mean(src, idx, dim=0).shape))
EOF
[ $? -ne 0 ] && die "torch_scatter GPU forward"

say "[9/11] torchsparse v2.1.0 + torch-2.9 patch"
# v2.1.0 uses Tensor.type(), removed in torch 2.9. Upstream master fixes it but
# adds a Rust/maturin dep that will not build on a compute node, so master's fix
# is applied by hand to the same three .cu files. No operator logic changes.
# torchsparse's CPU hashmap includes <google/dense_hash_map>, i.e. the sparsehash
# headers. On vulcan those came from `module load sparsehash` (that is what the line
# in scripts/vulcan/ENV_SETUP.md is for); star2 has no such module, so job 237206
# died immediately with
#   hashmap_cpu.hpp:7:10: fatal error: google/dense_hash_map: No such file or directory
# Provide it from conda-forge and put $CONDA_PREFIX/include on the C++ search path,
# since torchsparse's setup.py does not add it on its own.
if [ ! -f "$CONDA_PREFIX/include/google/dense_hash_map" ]; then
    echo "installing sparsehash headers (google/dense_hash_map) from conda-forge ..."
    conda install -y -q -c conda-forge sparsehash || die "sparsehash headers"
fi
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"
echo "dense_hash_map: $([ -f "$CONDA_PREFIX/include/google/dense_hash_map" ] && echo present || echo MISSING)"

if ! python -c "import torchsparse" 2>/dev/null; then
    mkdir -p $D/build_src && cd $D/build_src
    [ -d torchsparse ] || git clone -q https://github.com/mit-han-lab/torchsparse.git || die "torchsparse clone"
    cd torchsparse && git checkout -q 07f021b || die "torchsparse checkout"
    for f in $(grep -rl '\.type(), "' torchsparse/backend/ 2>/dev/null); do
        sed -i 's/\.type(), "/.scalar_type(), "/g' "$f"; echo "patched: $f"
    done
    echo "unpatched remaining: $(grep -rn '\.type(), "' torchsparse/backend/ 2>/dev/null | wc -l)  (expect 0)"
    pip install --no-build-isolation . > "$(build_log torchsparse)" 2>&1 || die "torchsparse build"
    cd "$ROOT"
fi
python - <<'EOF' || exit 1
import torch
from torchsparse import SparseTensor, nn as spnn
feats  = torch.randn(512, 32, device='cuda')
coords = torch.randint(0, 16, (512, 4), dtype=torch.int32, device='cuda'); coords[:, -1] = 0
x = SparseTensor(feats, coords, stride=1)
conv = spnn.Conv3d(32, 32, kernel_size=3, stride=1, dilation=1, bias=True).cuda()
print("TORCHSPARSE OK", tuple(conv(x).feats.shape))
EOF
[ $? -ne 0 ] && die "torchsparse GPU forward"

say "[10/11] nvdiffrast (module-level import at texgen_network.py:15)"
if ! python -c "import nvdiffrast.torch" 2>/dev/null; then
    mkdir -p $D/build_src && cd $D/build_src
    [ -d nvdiffrast ] || git clone -q https://github.com/NVlabs/nvdiffrast.git || die "nvdiffrast clone"
    pip install ./nvdiffrast > "$(build_log nvdiffrast)" 2>&1 || die "nvdiffrast build"
    cd "$ROOT"
fi
python -c "import nvdiffrast.torch as dr; print('NVDIFFRAST OK', dr.RasterizeCudaContext() is not None)" || die "nvdiffrast cuda context"

say "[11/11] full import smoke — the environment gate"
cd "$ROOT/TEXGen" || die "no TEXGen checkout at $ROOT/TEXGen"
python - <<'EOF' || exit 1
import importlib, sys
mods = ["torch", "pytorch_lightning", "torch_scatter", "torchsparse", "spconv.pytorch",
        "nvdiffrast.torch", "timm", "transformers", "diffusers", "omegaconf", "cv2",
        "jaxtyping", "wandb", "pandas", "pyarrow",
        "spuv", "spuv.systems.lightgen_system",
        "spuv.models.sparse_networks.lightgen_pointuvnet"]
bad = []
for m in mods:
    try:
        importlib.import_module(m); print("OK  ", m)
    except Exception as e:
        bad.append(m); print("FAIL", m, "->", repr(e)[:220])
print("FAILED:", bad)
sys.exit(1 if bad else 0)
EOF
[ $? -ne 0 ] && die "import smoke — see FAIL lines above"

mark "DONE spconv=${SPCONV_RESULT%% *} flash=${FLASH_RESULT%% *}"
say "BOOTSTRAP COMPLETE"
summary
echo "env: $D/miniconda3/envs/$ENV_NAME"
exit 0
