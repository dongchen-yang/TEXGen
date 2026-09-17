# TEXGen-Emission

TEXGen-Emission is our fork of TEXGen (SIGGRAPH Asia 2024, `CVMI-Lab/TEXGen`) for emission-map
generation: a 13-channel UV input (noisy emission 3, position 3, albedo 3, metallic 1, roughness
1, alpha 1, occupancy 1) at 256² goes through PointUVNet and comes out as an emission RGB
velocity, trained with flow matching, MSE+L1 on the velocity over the UV islands, and CLIP
conditioning on the shape's thumbnail with a fixed text prompt. Fork `dongchen-yang/TEXGen`,
branch `main`, directory `TEXGen-Emission/` in the parent repo. Parent repo conventions:
`lightgen/AGENTS.md`; the agent rule for this folder: `lightgen/.claude/rules/texgen_emission.md`.

**Where things are.** `spuv/` is upstream TEXGen edited on top. Upstream files that took our
role are renamed `texgen_emission_*` (`mesh_uv.py` keeps its name) and edited in place, with
upstream's class names; upstream files kept at their path carry only the fixes listed under
"The upstream delta"; our added helpers are under `spuv/utils/`. The files that matter:
`spuv/data/mesh_uv.py` (the atlas loader and datamodule, `MeshUVDataset`; upstream's Objaverse
loader is gone),
`spuv/systems/texgen_emission_base.py` (`TEXGenBaseSystem`: construction, EMA, the hooks) and
`spuv/systems/texgen_emission_test.py` (`TEXGenDiffusion`: our conditioning, loss, training,
`test_step` and validation, the Euler sampler),
`spuv/models/sparse_networks/texgen_emission_network.py` (`PointUVNet` on pre-baked UV maps),
`spuv/utils/uv_metrics.py` (the shared tensor helpers),
`spuv/utils/launch_ext.py` and `spuv/utils/wandb_utils.py` (what `launch.py` calls),
`spuv/utils/seed.py`, `spuv/utils/memory_tracker.py`. At the root: `launch.py` (upstream's
entry point plus a few lines), `inference_specific_samples.py`, `requirements.txt` (upstream's
list with three pins changed: `bitsandbytes`, `opencv-python` and `flash-attn`; it names no torch
version, and the `texgen` env runs torch 2.1.0 + cu118), `configs/`, `tools/`,
`tests/`.
Four small re-export files in `spuv/` keep the paths the published checkpoints need; do not
remove them. `spuv/data/lightgen_uv.py`, `spuv/systems/lightgen_system.py` and
`spuv/models/sparse_networks/lightgen_pointuvnet.py` re-export our classes under the names every
published `configs/parsed.yaml` uses. Two of the four are pickled paths as well: the published
checkpoint's pickle names `spuv.systems.lightgen_system.LightGenSystem` and
`spuv.systems.texgen_base.LossConfig` (the latter file re-exports `LossConfig` and nothing else),
so `torch.load` fails without either of them even if no config named them.

## The paper run

| | |
|---|---|
| display name | TEXGen-Emission |
| run | `texgen_alpha_74k_v2_agentic`, wandb project `LightGen` |
| config | `configs/texgen_emission.yaml` (was `configs/lightgen_pointuv_256_batch32_emissive_74k_v2_alpha_venus19_agentic.yaml`) |
| launcher | `deprecated/2026-09-16/alpha_agentic_paper_run/scripts/star2/train_74k_v2_alpha_agentic_venus19.sh`, unedited (it pins cs-venus-19 and the branch `texgen-74k-v2-venus05`); its env came from `bootstrap_texgen_bw.sh` beside it |
| code | commits `a72fa09`..`ae0e465` of branch `texgen-74k-v2-venus05`, tag `pre-trim-2026-09-16` |
| training | 4 × RTX PRO 6000 Blackwell, cs-venus-19 (job 248498, epochs 0–78) then cs-venus-05 (job 249206, resumed from epoch 74), env `texgen-bw` (torch 2.9 + cu128), 36,255 train shapes, 284 steps/epoch, 125 epochs, 35,500 steps |
| checkpoint | `epoch=124-step=35500.ckpt`, jupiter `outputs/texgen_alpha_74k_v2_agentic/ckpts/` (with `configs/parsed.yaml` one level up), local mirror `TEXGen-Emission/outputs/texgen_alpha_74k_v2_agentic/ckpts/` |
| inference | the parent bridge `evaluation/newdata_eval/run_texgen_infer.py` in the `texgen` env with `TEXGEN_ENABLE_FLASH=0`; test-set numbers in `docs/evaluation.md` § Test set |

## Train

`python launch.py --config configs/texgen_emission.yaml --gpu 0 --train [--wandb]` from this
folder, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The config sets
`auto_resume: true`, and `spuv/utils/launch_ext.py`'s `resolve_resume` then picks what a
`--train` launch resumes from, in this order: the run's own newest (by modification time)
`last*.ckpt` in `checkpoint.dirpath`, so a requeued run continues where it stopped; else the
path given as `resume=<path>`; else it starts fresh. It reads the chosen checkpoint once, on
every rank. Every checkpoint also stores the LR-scheduler state, restored at train start, and,
when the run logs to wandb, its run id, which a `--wandb` launch resumes; `wandb.dir` sets where
the run is written (it reaches `wandb.init` through the logger's `save_dir`, so an offline run no
longer lands in the current directory). `fit_with_graceful_exit` finishes the wandb run on
`KeyboardInterrupt` or `SystemExit` and then re-raises, so an interrupted launch exits non-zero
and the `trainer.test` after fit does not run; still judge a run finished by its checkpoint step,
not by its exit code. Under DDP, Lightning replaces the validation and test samplers, so each
rank scores its own shard of the split; `val/mse` and `val/psnr` are reduced across ranks
(`sync_dist=True`), while the previews that reach wandb are rank 0's shard. The config's data
paths and `checkpoint.dirpath` still name the paper run's node-local July-era root on star2.

The paper run's launcher and env bootstrap are its record, not live code:
`deprecated/2026-09-16/alpha_agentic_paper_run/scripts/star2/train_74k_v2_alpha_agentic_venus19.sh`
(it pins cs-venus-19, the branch `texgen-74k-v2-venus05` and the July-era root) and
`bootstrap_texgen_bw.sh` beside it (it built the node-local `texgen-bw` env). A node-agnostic
star2 launcher, which picks the first node that can take the run, on the existing shared env
`~/envs/texgen` on the star2 home (torch 2.1 + cu118, so no Blackwell nodes), comes with Tasks
13 and 14 of the restructure plan
(`lightgen/docs/superpowers/plans/2026-09-16-texgen-restructure.md`), which also rewrite this
section.

**The next run trains on the new data.** The current dataset is jupiter `datasets/dataset_73k`:
one folder per shape with `atlas.npz` (alpha inside as the eighth key; no sidecars). The loader
reads that layout as it is (`atlas.npz` first, `somage.npz` for the older staged roots). What a
run on it still needs, and what this restructure did not build: a parquet indexed by shape id
with `ditem_dir` and `success` for its shapes, a split JSON of positional indices into it, the
TexVerse thumbnails beside it as `thumbnails/<sha>.png`, and a staged copy on the training node.
The paper run trained on the July-era node root, patched with `alpha.npy` sidecars by
`deprecated/2026-09-16/alpha_sidecar_staging/stage_alpha_venus05.sh`; that script is not used
again.

## Infer

The supported path is the parent bridge:
`python evaluation/newdata_eval/run_texgen_infer.py --texgen_root <repo>/TEXGen-Emission --ckpt <run>/ckpts/<file>.ckpt --data_root <root> --parquet <parquet> --shas_file <ids> --seed N --out_dir <out>`,
which chdirs into the submodule and calls `inference_specific_samples.inference_samples`. Pass
`--texgen_root`: the bridge's default, `<repo>/TEXGen`, no longer exists after the rename (the
evaluation lane will fix it). The same call is available directly as
`python inference_specific_samples.py --ckpt … --data_root … --parquet … --shas_file … --out_dir … --seed N`
from this folder. The model config is the run's `configs/parsed.yaml` beside the checkpoint's
`ckpts/` directory, not `configs/texgen_emission.yaml`, and it samples with the EMA weights when
that config enables them (the paper run's does). Per shape it
writes `input_albedo.png`, `gt_emission.png`, `pred_emission.png`, `mask.png`, `thumbnail.png`
(when the shape has one) and `comparison.png`. The noise for shape `sha` at seed `N` is
`torch.manual_seed(sample_seed(sha, N))` set right before `test_pipeline`; `sample_seed` in
`spuv/utils/seed.py` is a verbatim copy of the one in the parent's
`evaluation/newdata_eval/seedutil.py`, and `tests/test_seed.py` checks that the two return the
same values.

**The flash gate.** `TEXGEN_ENABLE_FLASH=0` selects the dense-attention branch of the PTv3
blocks. The published checkpoint trained on that branch (set to 0 for comparability with the two
earlier alpha runs; flash-attn does import on sm_120), and running it with the flash branch active
gives different numbers with no error. Every inference of this checkpoint sets the variable; the
parent's job scripts and the cs-venus-05 driver export it, and `tests/test_flash_gate.py` pins the
parsing.

## Tests

`cd TEXGen-Emission && conda run -n texgen bash -c 'python -m pytest tests -q'` (pytest was added
to the `texgen` env on 2026-09-16): 64 passed with the GPU. `test_channel_layout.py`,
`test_flash_gate.py` and the backbone check in `test_config_resolution.py` import the network,
and `torchsparse` initializes CUDA when it is imported, so they skip without a CUDA device (60
passed, 3 skipped); run the suite with `CUDA_VISIBLE_DEVICES=` while the GPU is busy. Those counts
assume the published run's `parsed.yaml` is mirrored under
`outputs/texgen_alpha_74k_v2_agentic/configs/` and this folder sits inside the parent repo;
otherwise the tests that read them skip. What each file pins:

- `test_data_contract.py`: the atlas loader's keys, shapes and value ranges, the alpha rules (the
  npz key wins over the sidecar, missing alpha raises at construction), split indices read as
  positions into the success-filtered rows, and the current bake's layout (`atlas.npz` with alpha
  inside and no sidecar, and `atlas.npz` winning over an older `somage.npz`);
- `test_channel_layout.py` (GPU): a 13-channel input gives a 3-channel velocity, and a 12-channel
  batch against the 13-channel config fails loudly;
- `test_config_resolution.py`: the config's class strings and the alpha guard; the published
  `parsed.yaml` resolving through the re-exports to the same classes and still loading through
  `load_config`; the backbone class (GPU);
- `test_system_classes.py`: the two pickled paths (`LossConfig` and the `lightgen_system`
  re-export), the step methods Lightning calls and the test hooks that alias the validation ones,
  the check that rejects a nonzero loss weight the loss does not implement (on the live config and
  on every mirrored `parsed.yaml`), the x0 the training panel derives from the flow-matching
  target, and the image-log override's contract;
- `test_seed.py`: parity with the parent's `sample_seed` and a pinned value;
- `test_flash_gate.py` (GPU): the parsing of `TEXGEN_ENABLE_FLASH`;
- `test_uv_metrics.py`: the shared tensor helpers;
- `test_launch_ext.py`: the resume choice and the metadata it reads, the checkpoint-dir helpers,
  the wandb logger arguments (the run directory among them) and stale-cache cleanup, the shutdown
  handlers, the re-raise on an interrupt and the `torch.load` patch;
- `test_training_hooks.py`: the training hooks the fold moved into `TEXGenBaseSystem`, driven on
  the CPU with the toy backbone and tokenizer of `tests/toy_modules.py` — four synthetic shapes,
  fit, resume, then `trainer.test`; one EMA update per batch and the count restored on resume, the
  EMA weights swapped in for validation and test and the training weights restored after,
  `_scheduler_states` written to the checkpoint and read back, and each moved hook still a plain
  function on the class (a property would pass an `in __dict__` check and never be called).

## Data contract

`spuv/data/mesh_uv.py` reads one npz per shape, `atlas.npz` (the current bake) or
`somage.npz` (the July-era staged roots), with keys `occupancy` bool, `position` and
`objnormal` uint16, `color`, `metal`, `rough`, `emission_color` uint8, all `[H, W, C]` at 512²,
plus alpha from the npz key `alpha` (the current bake) or an `alpha.npy` sidecar (the July-era
roots, including the eval root the published numbers came from), downsamples
to the config's 256² (bilinear for continuous maps, nearest for the occupancy), normalizes
emission to [−1, 1], and reads the CLIP thumbnail from `<data_root>/thumbnails/<sha>.png`. Every
shape in a split needs its thumbnail. `collate_fn` takes the batch's thumbnail from its first
shape: if that one is missing, the whole batch is conditioned on the albedo UV map, with a
warning, and a thumbnail missing for any later shape fails in `torch.stack`. The shape id is the
parquet's index, and the shape's folder is `<data_root>/<ditem_dir>`. A run with
`use_alpha: true` and no alpha fails with
`AlphaUnavailable`, never silently: the first shape is checked at construction, and the loader
re-raises the error for any later shape. Split JSONs hold positional indices into the parquet
filtered to `success == True`. The bake side of the contract is
`data_processing/data_preparation/README.md`.

## Measured findings worth knowing

- **`save_top_k` with `monitor: val/psnr` can stop checkpointing entirely.** With a metric that
  peaks early, top-k keeps its early files and later epochs are never saved; the run of record
  uses `save_top_k: -1` and `every_n_epochs: 5`.
- **`val/psnr` rewards a darker prediction.** Measured on `texgen_alpha_74k_v2` 2026-08-13: epoch
  4 beat epoch 24 by 0.80 dB on PSNR while emitting 55% of the ground truth's light against 79%;
  checkpoints are selected on the point-sampled evaluator in the parent repo, not on this metric.
- **`shuffle_orders=True` consumes the CPU RNG at inference**, hundreds of `torch.randperm`
  calls per shape from the serialization step, and the permutation changes the output. That is
  why the seed is a global `torch.manual_seed`, never a `torch.Generator`.
- **The H100 VRAM model underestimates on the torch 2.9 stack.** The fir probe's model (about
  14 GB static plus 1.37 GB per sample) predicted 35.9 GB for micro-batch 16 on 4 × L40S, which
  peaked at 41.99 GB and ran out of memory on 44.39 GiB usable (vulcan, job 217827, 2026-07-29;
  micro-batch 8 × accumulation 4 trained). On the Blackwell cards per-GPU batch 32 peaked at
  69.2 GB of 95.6 (cs-venus-05). Measure on the target stack before sizing a batch.

## The upstream delta

`bash tools/upstream_diff.sh` prints it: `git diff --stat` of `HEAD` against
`upstream/main` for `launch.py`, `spuv/`, `requirements.txt` and `.gitignore`, then the three
renamed files, each against its upstream source at `HEAD`, paired explicitly by name in the
script. (In the first part git pairs only `texgen_network.py` by itself; `texgen_base.py` still
exists as the re-export, and `texgen_test.py` shows as deleted.)

- **Upstream files kept at their path, each change with a reason.** `launch.py` (11 lines added,
  5 removed: the two environment variables the fork sets, the calls into
  `spuv/utils/launch_ext.py` and `wandb_utils.py`, and the checkpoint dirpath and kwargs read
  through `launch_ext`); `spuv/systems/base.py` (one `field(default_factory=…)`, a Python 3.11
  rule); `spuv/models/tokenizers/clip.py` (the SD-3.5-large tokenizer and text encoder, `no_grad`
  image embeds, the 768-dim comments); `spuv/utils/config.py` (`auto_resume`, `wandb`, and
  `custom_output_dir`, which every published `parsed.yaml` carries and `parse_structured` would
  reject if removed); `spuv/utils/ops.py` and `misc.py` (the `torch.amp` API wrappers and
  `weights_only=False` in `load_module_weights`); `requirements.txt` pins; `.gitignore` (our
  output and log dirs and `.claude/`; upstream's `*.sh` rule is dropped). `spuv/utils/typing.py`
  and `spuv/utils/saving.py` are upstream's, unchanged.
  Outside the script's paths, upstream's `configs/texgen_test.yaml` was removed by an earlier
  cleanup (`39c0452`); `README.md`, `assets/`, `static/` and upstream's files in `tools/` are
  unchanged.
- **Upstream files that became ours.** `spuv/data/mesh_uv.py` (our atlas loader replaced the
  Objaverse loader, in a commit of its own, and was edited in the next).
  `spuv/systems/texgen_emission_base.py`, `spuv/systems/texgen_emission_test.py`
  and `spuv/models/sparse_networks/texgen_emission_network.py`, renamed from `texgen_base.py`,
  `texgen_test.py` and `texgen_network.py` in commits of their own and edited in the next, so
  `git log --follow` shows every edit. In the network file those edits are the pre-baked
  `forward`, the adaptive-skip in-channel count read from config (a bug for any input width but
  10), CLIP dims 1024 → 768 for the SD-3.5 text encoder, and the `TEXGEN_ENABLE_FLASH` env gate.
  `spuv/systems/texgen_base.py` is now the pickled-path re-export.
- **Ours, added.** `spuv/utils/{uv_metrics,launch_ext,wandb_utils,seed,memory_tracker}.py` and the
  three `lightgen_*` re-exports.
- **Upstream code nothing live imports.** `spuv/models/sparse_networks/utils/feature_baking.py`
  (the render-and-bake path; its `__main__` block names the removed `ObjaverseDataModule`),
  `spuv/utils/nvdiffrast_utils.py` and `spuv/utils/rasterize.py`, `spuv/models/camera.py`,
  `isosurface.py`, `lpips.py`, `networks.py`, `perceptual_loss.py`, `timestep.py`, and
  `spuv/utils/image_metrics.py` and `snr_utils.py`, kept as upstream left them.
  `spuv/models/renderers/rasterize.py` is not in that list: the backbone imports
  `NVDiffRasterizerContext` from it and `PointUVNet.configure` builds one on every backbone build
  (upstream's `ctx=self.ctx` passthrough), so `nvdiffrast` has to import, and build its plugin, on
  any node that runs the model.

## History

`deprecated/2026-09-16/` holds everything the 2026-09-16 trim moved out of the live tree
(`deprecated/README.md` lists the folders and which of them mirror the pre-trim paths): the v1
somage-era lineage (the 1k baseline, the vanilla / GT-mask / mask-classification variants and
their overfit configs, `filter_*.py`, `precompute_clip_embeddings.py`, the xgutils render
scripts, `slurm_train*.sh`, the fir wrappers), the 12-channel `texgen_vanilla_74k_v2` lane with
its venus05 and vulcan launchers and the vulcan env recipe, the alpha runs without a paper row
(`texgen_alpha_74k_v2`, `_nonzero`, `_nonzero_nocopy`, the fir port of `_agentic`), the alpha
sidecar staging script, the paper run's own launcher and its `texgen-bw` bootstrap, the early
`simple_uv_unet.py`, a `uv init` stub (`main.py`, `pyproject.toml`), the old docs with the
2026-03/04 experiment log. The two filtering rows of `docs/evaluation.md` (`texgen_alpha_74k_v2`,
`_nonzero_nocopy`) cite the folder. Tag `pre-trim-2026-09-16` (= the head of
`texgen-74k-v2-venus05`, `ae0e465`) is the last commit before the trim and holds the in-file
code the fold dropped (the mask-only mode, the GT-mask oracle, the mask-classification and
dark-region losses, the precomputed-CLIP path, the disabled 3D-render block, the DDPM noise
schedule this system never read). Tags `archive/texgen-74k-v2-venus05` and
`archive/texgen-74k-v2-vulcan` mark the two old branch heads;
the branches stay on origin, frozen, and the local `texgen-74k-v2-venus05` ref stays until the
evaluation lane repoints the cs-venus-05 driver.

## Verification (2026-09-16)

Reproduction gate: the published checkpoint, the 204 agentic_clean test shapes, seed 0, the
workstation 4090, `texgen` env, `TEXGEN_ENABLE_FLASH=0`; `pred_emission.png` byte-compared with
the parent's `evaluation/newdata_eval/compare_pred_dirs.py`. Two runs of the same code in
PyTorch's normal mode do not reproduce each other byte for byte, so the gates run in
deterministic mode (`torch.use_deterministic_algorithms(True, warn_only=True)`,
`torch.backends.cudnn.deterministic = True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, set from outside
the repo), which does. A gate passes only at 204 / 0 / 0 against det1.

| pair | same | different | missing |
|---|---|---|---|
| before (tag `pre-trim-2026-09-16`, normal mode) ↔ the published seed-0 test predictions | 16 | 188 | 0 |
| before ↔ before, run again (normal mode) | 121 | 83 | 0 |
| det1 ↔ det2 (tag `pre-trim-2026-09-16`, deterministic mode, back to back) | 204 | 0 | 0 |
| det1 ↔ after the trim (`[trim commit]`, from a detached worktree) | 204 | 0 | 0 |
| det1 ↔ after the fold and the rename (`[fold commit]`, `TEXGen-Emission/`) | [same] | [different] | [missing] |
| det1 ↔ after the review fixes (`[final commit]`) | [same] | [different] | [missing] |

The normal-mode differences are float noise: at most 1 LSB on at most 0.42% of pixels between the
two local runs, at most 3 LSB on under 0.7% of pixels against the published run. Deterministic
mode shifts the outputs against normal mode by at most 4 LSB on at most 0.68% of pixels, and it
applies to the before and the after runs alike; the gate proves the code did not move and does
not re-derive the published numbers.

Training was not re-run on the old data: verification of the trimmed code is inference only, and
no test drives the training path on real data. What does cover it:
`tests/test_launch_ext.py` pins the resume, wandb and shutdown helpers;
`tests/test_system_classes.py` pins the class identities, the step methods, the loss guard and
the training panel's x0; `tests/test_training_hooks.py` runs a toy CPU fit, resume and
`trainer.test` and checks the EMA count, the EMA swap and restore, the scheduler state in the
checkpoint, and that every moved hook is still a function. Beyond the tests, the training path
was read hunk by hunk against the pre-trim tag and driven through a CPU training smoke (a toy
backbone and tokenizer, synthetic shapes, fit → resume → test): its Lightning `metrics.csv` and
all 32 saved previews are byte-identical between tag `pre-trim-2026-09-16` and `3312ebd`, and
after the review fixes differ only where those fixes intended (below). The first launch of the
retrain on the new data is the real training-path check.

Inference is proven byte for byte; training numerics are equivalent but not bitwise reproducible
against the paper run. Dropping the construction this fork never used (LPIPS, the DDIM test
scheduler, the system-level rasterizer, the second CLIP build) changes how much CPU randomness is
drawn after `seed_everything`, so a rerun of the paper config on this code does not follow the
paper run's trajectory step for step, even in deterministic mode.

Tests: 64 passed with the GPU; 60 passed and 3 skipped on the CPU. Upstream delta after the
fold (`bash tools/upstream_diff.sh`):

[the script's output at the final commit, in a code block]

### Training-side differences from the paper run's code

None of them changes the weights a training step produces. They matter when a new run's logs,
memory figures or wandb panels are read beside the paper run's.

- Each validation batch cleans up memory twice instead of three times (upstream's
  `on_validation_batch_end` and one call in `validation_step`; the fork had a third inside its
  own `test_step`), and the `after_validation` log is now taken before the first cleanup.
- The `train/predictions` panel no longer carries the "GT Emission Mask (condition)" image: the
  loader does not emit `gt_emission_mask`.
- Construction no longer builds the DDIM test scheduler, the DDPM noise schedule, a system-level
  `NVDiffRasterizerContext`, `LPIPS`, SSIM or PSNR, and it builds the CLIP tokenizer once instead
  of twice. None of them held parameters or buffers, so the checkpoint's keys and the optimizer's
  parameter groups are unchanged.
- With `--benchmark`, the `backward` and `train_batch_end` timers no longer include the memory
  logs and the 10-batch cleanup, which now sit outside the timed region.
- A resumed run uses less host memory per rank: the checkpoint is read once, inside
  `resolve_resume`, and freed when it returns; the fork read it twice in `main` and held the
  second copy across `fit`.
- `trainer.test` after fit runs on the EMA weights, as upstream's `ema_scope` did.
- `val/mse` and `val/psnr` are logged with `sync_dist=True`.
- The training panel's x0 is derived from the flow-matching interpolation the loss uses; the old
  formula was the DDPM one and drew roughly the noisy input.
- `wandb.dir` reaches `wandb.init`, so an offline run is written where the config says.
- An interrupted launch exits non-zero.
- The resume order (the run's own newest `last*.ckpt` before any explicit `resume=<path>`) and
  one checkpoint read per rank instead of two.
