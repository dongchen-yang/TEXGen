# Deprecated / archived TEXGen LightGen material

Moved here during the 2026-07-09 TEXGen cleanup (Phase 3). Prefer the active
configs and scripts at the repo root / `configs/` / `scripts/fir/`. See
[`../LIGHTGEN_WORKFLOW.md`](../LIGHTGEN_WORKFLOW.md) for the live map.

## Layout

| Path | Contents |
|------|----------|
| `configs/` | Overfit10 / ablation / lambda01–lambda10 sweep YAMLs superseded by the three baseline overfits + lambda001 recipes |
| `docs/` | Stale setup docs (`START_TRAINING`, `TRAIN_FULL_DATASET`, `WANDB_SETUP`, old `LIGHTGEN_README`) |
| `scripts/local/` | Workstation overfit10 launchers |
| `scripts/star2/` | One-off star2 overfit10 launchers |
| `scripts/root/` | Root sweep/overfit/setup helpers no longer on the production path |
| `comparisons/` | Overfit comparison scripts + generated PNGs |
| `utils/` | Ad-hoc verify/metrics helpers and `inference_custom_thumbnail.py` |

Nothing here is required for fir 74k training or the three active baseline variants.
Paths under Jupiter (`/cs/3dlg-jupiter-project/lightgen/`) still hold wandb/outputs backups.

## 2026-09-16 — trimmed to the paper's run

Everything under `2026-09-16/` was live until 2026-09-16 and was moved here when the submodule
was trimmed to the run the paper uses (`texgen_alpha_74k_v2_agentic`; spec
`lightgen/docs/superpowers/specs/2026-09-16-texgen-restructure-design.md`). The pre-trim head is
tag `pre-trim-2026-09-16`. Paths inside the lineage folders `v1_somage_era/`,
`v2_vanilla_no_alpha/`, `alpha_no_paper_row/` and `alpha_agentic_paper_run/` mirror the pre-trim
tree; `alpha_sidecar_staging/` and the top level hold their files flat. The 2026-07-09 section's
pointers (`scripts/fir/`, `../LIGHTGEN_WORKFLOW.md`) predate this trim; both moved into
`2026-09-16/`.

| folder | what | runs and rows |
|---|---|---|
| `v1_somage_era/` | the 1k baseline on somage data: the vanilla / GT-mask / mask-classification configs and their overfit configs, the fir wrappers, the somage filtering and CLIP precompute one-offs, the fir 74k launchers, the xgutils render scripts (do not run) | `lightgen/docs/status/evaluation-results.md` (closed 2026-08-18) |
| `v2_vanilla_no_alpha/` | the 12-channel `texgen_vanilla_74k_v2` lane: its config and cs-venus-05 launcher, the vulcan fallback with its env recipe | `lightgen/docs/evaluation.md` § Other models |
| `alpha_no_paper_row/` | the alpha runs without a paper row: `texgen_alpha_74k_v2` (unfiltered), `_nonzero`, `_nonzero_nocopy` (heuristic filtering), the fir port of `_agentic` | `lightgen/docs/evaluation.md` § The table and § Data filtering |
| `alpha_sidecar_staging/` | `stage_alpha_venus05.sh` (was `scripts/star2/`): patched the July-era node-local training root with `alpha.npy` sidecars beside each `somage.npz`. The current bake (jupiter `datasets/dataset_73k`) stores alpha inside `atlas.npz`, so new runs stage the dataset directly and this script is not used | the paper run's training root (history) |
| `alpha_agentic_paper_run/` | `scripts/star2/train_74k_v2_alpha_agentic_venus19.sh`, the launcher the paper run (`texgen_alpha_74k_v2_agentic`) trained with on cs-venus-19 and resumed on cs-venus-05, and `scripts/star2/bootstrap_texgen_bw.sh`, the node-local `texgen-bw` env builder it ran under. Unedited: they pin a node, the branch `texgen-74k-v2-venus05` and a per-node env, and they check the July-era training root (its parquet md5 and the `alpha.npy` sidecars). A node-agnostic launcher on one shared star2 env replaces them (`../TEXGEN_EMISSION.md` § Train) | the paper run: `lightgen/docs/evaluation.md` § Test set, `lightgen/docs/status/checkpoints.md` § 2026-08-23 |
| `env_records/` | `requirements_125.txt` (was the submodule root): a torch 2.9.0 + cu126 package list that matches the workstation env `texgen_125`. It is neither the `texgen` env this repo runs (torch 2.1.0 + cu118, pinned by `../requirements.txt`) nor the paper run's `texgen-bw` (cu128, built by `alpha_agentic_paper_run/scripts/star2/bootstrap_texgen_bw.sh`), and nothing here reads it | |
| top level | `simple_uv_unet.py` (was `spuv/models/`; an early plain UV U-Net, never trained for a published number), `main.py` + `pyproject.toml` (a `uv init` stub), `LIGHTGEN_WORKFLOW.md` and `CLAUDE_2026-09-07.md` (the old docs; their live content is in `../TEXGEN_EMISSION.md`, the 2026-03/04 experiment log stays in `CLAUDE_2026-09-07.md`); the others were at the submodule root, `CLAUDE_2026-09-07.md` as `CLAUDE.md` | |
