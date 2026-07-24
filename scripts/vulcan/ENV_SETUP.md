# TEXGen env on vulcan (L40S / sm_89) — working recipe

Built 2026-07-23 at `/scratch/dya78/lightgen_repo/env`. Reproduce with the two
scripts in this directory; the ordering below is load-bearing — deviating from it
reproduces the failures listed at the bottom.

## Final versions (all 13 imports verified)

| package | version | source |
|---|---|---|
| torch | 2.9.0+computecanada | wheelhouse (`--no-index`) |
| torchvision | 0.24.1+d801a34 | **source build** (no compatible wheel) |
| torchsparse | 2.0.0b (v2.1.0 tree, patched) | **source build + patch** |
| flash-attn | 2.7.3 | **source build** (GPU node, ~1 h) |
| pointops, nvdiffrast | 1.0 / 0.4.0 | source build |
| torch-scatter/cluster/sparse | +torch29.computecanada | wheelhouse |
| spconv | 2.3.8+computecanada | wheelhouse |
| lightning | 2.5.0+computecanada | wheelhouse |
| transformers / diffusers | 4.28.1 / 0.28.0 | wheelhouse / PyPI |
| setuptools | **69.5.1** | PyPI (see gotcha 1) |
| numpy | 1.26.4+computecanada | wheelhouse |

## Order

1. `virtualenv --no-download env` under `module load StdEnv/2023 cuda/12.6 gcc
   opencv arrow/22.0.0 sparsehash python/3.11.5`.
2. Wheelhouse layer (`--no-index`): torch==2.9.0, torch-scatter/cluster/sparse,
   torch-geometric, spconv, lightning, numpy, scipy, open3d.
3. PyPI layer: setuptools==69.5.1, wheel, transformers, diffusers,
   huggingface_hub==0.25.2, accelerate, wandb, lpips **--no-deps**, packaging<25.
4. Source builds on a **GPU node** (`--gres=gpu:l40s:1`), `CUDA_HOME` exported,
   `TORCH_CUDA_ARCH_LIST=8.9`: torchvision, torchsparse (patched), pointops,
   nvdiffrast, flash-attn.

## Gotchas that cost a full day

1. **Broken setuptools metadata is the #1 trap.** If `python -c "import
   setuptools; print(setuptools.__version__)"` prints `0.dev0+unknown`, the
   `dist-info` directory is missing — the package files are present, so pip
   thinks it is installed and uninstall/reinstall does nothing. Every source
   build then dies with `error: invalid command 'dist_info'`. Fix:
   ```bash
   SP=$VIRTUAL_ENV/lib/python3.11/site-packages
   rm -rf $SP/setuptools $SP/setuptools-* $SP/pkg_resources $SP/_distutils_hack $SP/distutils-precedence.pth
   pip install --index-url https://pypi.org/simple --ignore-installed "setuptools==69.5.1"
   ```
   Then restore what the purge took with it: sympy, mpmath, networkx, filelock,
   fsspec, jinja2, typing-extensions. `pip check` enumerates the rest.

2. **`pip install lpips` silently upgrades torch.** Its dependency chain pulls
   torchvision from the wheelhouse, which pulls torch 2.13 — undoing the 2.9
   pin and breaking every `+torch29` extension wheel with
   `undefined symbol: torch::autograd::deleteNode`. Install it `--no-deps`.

3. **torchsparse version bind.** v2.1.0 uses `Tensor.type()`, removed in torch
   2.9 (`no suitable conversion from at::DeprecatedTypeProperties`). master
   fixes it (commit 385f5ce) but adds a Rust/maturin dependency (`cachebox`)
   that cannot build on a compute node. Solution: check out v2.1.0 (`07f021b`)
   and apply master's fix by hand — 3 `.cu` files, `.type(), "` →
   `.scalar_type(), "` (`scripts/vulcan/patch_torchsparse.sh`). No operator
   logic changes.

4. **`#!/bin/bash` batch scripts have no `module` function** on vulcan. Use
   `#!/bin/bash -l` or the whole module stack silently no-ops and `CUDA_HOME`
   comes out empty.

5. **flash-attn is NOT optional for TEXGen.** `ptv3_model_texgen.py` guards the
   import with try/except, but `texgen_network.py:753` hardcodes
   `enable_flash=True` and the block asserts on it. Budget ~1 h of GPU-node
   compile time (`MAX_JOBS=8`).

6. **Login-node SSH gets cut mid-install.** Run long installs inside a
   vulcan-side `tmux` session, not over the ssh connection.
