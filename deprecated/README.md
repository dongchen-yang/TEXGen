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
