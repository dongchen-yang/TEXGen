"""Compare 3 overfit experiment runs: Vanilla, GT Mask Cond, Mask Cls Loss."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────
RUNS = {
    "Vanilla": "../outputs/overfit_vanilla/lightgen/pointuv_overfit/tb_logs/version_0",
    "GT Mask Cond": "../outputs/overfit_gt_mask_cond/lightgen/pointuv_overfit_gt_mask_cond/tb_logs/version_0",
    "Mask Cls": "../outputs/overfit_mask_cls/lightgen/pointuv_overfit_mask_cls/tb_logs/version_0",
}

SAVE_DIRS = {
    "Vanilla": "../outputs/overfit_vanilla/lightgen/pointuv_overfit/save",
    "GT Mask Cond": "../outputs/overfit_gt_mask_cond/lightgen/pointuv_overfit_gt_mask_cond/save",
    "Mask Cls": "../outputs/overfit_mask_cls/lightgen/pointuv_overfit_mask_cls/save",
}

COLORS = {"Vanilla": "#1f77b4", "GT Mask Cond": "#ff7f0e", "Mask Cls": "#2ca02c"}
OUT_DIR = "comparison_overfit"
os.makedirs(OUT_DIR, exist_ok=True)

MILESTONE_ITERS = [0, 1000, 5000, 10000]


# ── 1. Extract TB scalars ──────────────────────────────────────────────────
def extract_scalars():
    """Extract val/psnr and val/mse from all runs."""
    data = {}
    for name, path in RUNS.items():
        ea = EventAccumulator(path)
        ea.Reload()
        data[name] = {}
        for tag in ["val/psnr", "val/mse"]:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[name][tag] = (steps, values)
        print(f"  {name}: {len(data[name]['val/psnr'][0])} val points")
    return data


def plot_training_curves(data):
    """Plot val/psnr and val/mse curves for all 3 runs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PSNR
    ax = axes[0]
    for name in RUNS:
        steps, values = data[name]["val/psnr"]
        ax.plot(steps, values, label=name, color=COLORS[name], alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Validation PSNR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MSE
    ax = axes[1]
    for name in RUNS:
        steps, values = data[name]["val/mse"]
        ax.plot(steps, values, label=name, color=COLORS[name], alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.set_title("Validation MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def print_metrics_table(data):
    """Print final and best metrics for each run."""
    print("\n" + "=" * 70)
    print(f"  {'Variant':<16} | {'Best PSNR':>10} @ {'Step':>6} | {'Final PSNR':>11} | {'Best MSE':>10} | {'Final MSE':>10}")
    print("  " + "-" * 66)
    for name in RUNS:
        steps_p, vals_p = data[name]["val/psnr"]
        steps_m, vals_m = data[name]["val/mse"]
        best_psnr_idx = np.argmax(vals_p)
        best_mse_idx = np.argmin(vals_m)
        print(
            f"  {name:<16} | {vals_p[best_psnr_idx]:>8.2f}dB @ {steps_p[best_psnr_idx]:>5} | "
            f"{vals_p[-1]:>9.2f}dB | {vals_m[best_mse_idx]:>10.6f} | {vals_m[-1]:>10.6f}"
        )
    print("=" * 70)


# ── 2. Visual comparison at milestones ─────────────────────────────────────
def create_visual_comparison():
    """Side-by-side composite comparison — one row per variant, one column per milestone."""
    run_names = list(SAVE_DIRS.keys())
    n_runs = len(run_names)
    n_iters = len(MILESTONE_ITERS)

    # Layout: rows = variants, cols = milestone iters
    # Composite is 992x256 (≈4:1), so give each cell generous width
    cell_w, cell_h = 8, 2.2
    fig, axes = plt.subplots(
        n_runs, n_iters,
        figsize=(cell_w * n_iters, cell_h * n_runs + 1),
    )

    for row, name in enumerate(run_names):
        for col, it in enumerate(MILESTONE_ITERS):
            ax = axes[row, col]
            img_path = os.path.join(
                SAVE_DIRS[name], f"it{it}-test", "preview", "composite_0_0_0.jpg"
            )
            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"iter {it}", fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(name, fontsize=13, fontweight="bold")

    fig.suptitle("Composite — Overfit Comparison", fontsize=15, fontweight="bold", y=1.0)
    plt.subplots_adjust(wspace=0.05, hspace=0.08)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "visual_composite.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 3. Convergence analysis ───────────────────────────────────────────────
def convergence_analysis(data):
    """Find when each variant reaches 90% of its best PSNR."""
    print("\n" + "=" * 70)
    print("  CONVERGENCE ANALYSIS")
    print("  " + "-" * 66)

    for name in RUNS:
        steps, values = data[name]["val/psnr"]
        best_psnr = max(values)
        threshold_90 = 0.9 * best_psnr

        # Find first step where PSNR >= 90% of best
        step_90 = None
        for s, v in zip(steps, values):
            if v >= threshold_90:
                step_90 = s
                break

        print(f"  {name:<16} | Best PSNR: {best_psnr:.2f}dB | 90% threshold: {threshold_90:.2f}dB | Reached at step: {step_90}")

    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[1/3] Extracting TB scalars...")
    data = extract_scalars()
    plot_training_curves(data)
    print_metrics_table(data)

    print("\n[2/3] Creating visual comparisons...")
    create_visual_comparison()

    print("\n[3/3] Convergence analysis...")
    convergence_analysis(data)

    print(f"\nAll outputs saved to: {OUT_DIR}/")
