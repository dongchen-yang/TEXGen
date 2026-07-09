"""Ablation study: leave-one-out on vanilla allreg (10-sample overfit).
Plots all 6 runs: no-reg, allreg, and 4 ablations (no_wd, no_ema, no_dropout, no_conddrop)."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Config ──────────────────────────────────────────────────────────────────
# TB log paths — ablation runs may need version_0 or version_1, auto-detect below
RUNS = {
    "Vanilla (no-reg)":       "../outputs/overfit10_vanilla/lightgen/pointuv_overfit10_vanilla/tb_logs",
    "Vanilla (allreg)":       "../outputs/overfit10_vanilla_allreg/lightgen/pointuv_overfit10_vanilla_allreg/tb_logs",
    "allreg − wd":            "../outputs/overfit10_vanilla_allreg_no_wd/lightgen/pointuv_overfit10_vanilla_allreg_no_wd/tb_logs",
    "allreg − EMA":           "../outputs/overfit10_vanilla_allreg_no_ema/lightgen/pointuv_overfit10_vanilla_allreg_no_ema/tb_logs",
    "allreg − dropout":       "../outputs/overfit10_vanilla_allreg_no_dropout/lightgen/pointuv_overfit10_vanilla_allreg_no_dropout/tb_logs",
    "allreg − cond_drop":     "../outputs/overfit10_vanilla_allreg_no_conddrop/lightgen/pointuv_overfit10_vanilla_allreg_no_conddrop/tb_logs",
}

COLORS = {
    "Vanilla (no-reg)":       "#888888",
    "Vanilla (allreg)":       "#1f77b4",
    "allreg − wd":            "#d62728",
    "allreg − EMA":           "#ff7f0e",
    "allreg − dropout":       "#2ca02c",
    "allreg − cond_drop":     "#9467bd",
}

LINESTYLES = {
    "Vanilla (no-reg)":       "--",
    "Vanilla (allreg)":       "-",
    "allreg − wd":            "-",
    "allreg − EMA":           "-",
    "allreg − dropout":       "-",
    "allreg − cond_drop":     "-",
}

LINEWIDTHS = {
    "Vanilla (no-reg)":       1.5,
    "Vanilla (allreg)":       2.5,
    "allreg − wd":            2.0,
    "allreg − EMA":           2.0,
    "allreg − dropout":       2.0,
    "allreg − cond_drop":     2.0,
}

SAVE_DIRS = {
    "Vanilla (no-reg)":       "../outputs/overfit10_vanilla/lightgen/pointuv_overfit10_vanilla/save",
    "Vanilla (allreg)":       "../outputs/overfit10_vanilla_allreg/lightgen/pointuv_overfit10_vanilla_allreg/save",
    "allreg − wd":            "../outputs/overfit10_vanilla_allreg_no_wd/lightgen/pointuv_overfit10_vanilla_allreg_no_wd/save",
    "allreg − EMA":           "../outputs/overfit10_vanilla_allreg_no_ema/lightgen/pointuv_overfit10_vanilla_allreg_no_ema/save",
    "allreg − dropout":       "../outputs/overfit10_vanilla_allreg_no_dropout/lightgen/pointuv_overfit10_vanilla_allreg_no_dropout/save",
    "allreg − cond_drop":     "../outputs/overfit10_vanilla_allreg_no_conddrop/lightgen/pointuv_overfit10_vanilla_allreg_no_conddrop/save",
}

MILESTONE_ITERS = [0, 1000, 5000, 10000]

OUT_DIR = "comparison_overfit10_ablation"
os.makedirs(OUT_DIR, exist_ok=True)


def find_best_version(tb_logs_dir):
    """Find the TB version with the most val/psnr data points."""
    if not os.path.exists(tb_logs_dir):
        return None
    best_path, best_count = None, 0
    for ver in sorted(os.listdir(tb_logs_dir)):
        ver_path = os.path.join(tb_logs_dir, ver)
        if not os.path.isdir(ver_path):
            continue
        try:
            ea = EventAccumulator(ver_path)
            ea.Reload()
            events = ea.Scalars("val/psnr")
            if len(events) > best_count:
                best_count = len(events)
                best_path = ver_path
        except Exception:
            continue
    return best_path


def extract_scalars():
    """Extract val/psnr and val/mse from all runs, auto-detecting best TB version."""
    data = {}
    for name, tb_dir in RUNS.items():
        path = find_best_version(tb_dir)
        if path is None:
            print(f"  SKIP (not found): {name} -> {tb_dir}")
            continue
        ea = EventAccumulator(path)
        ea.Reload()
        data[name] = {}
        for tag in ["val/psnr", "val/mse"]:
            try:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                data[name][tag] = (steps, values)
            except KeyError:
                data[name][tag] = ([], [])
                print(f"  WARNING: {name} missing tag {tag}")
        print(f"  {name}: {len(data[name]['val/psnr'][0])} val points (from {os.path.basename(path)})")
    return data


def plot_training_curves(data):
    """Plot val/psnr and val/mse for all 6 runs."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # PSNR
    ax = axes[0]
    for name in RUNS:
        if name not in data:
            continue
        steps, values = data[name]["val/psnr"]
        if not steps:
            continue
        ax.plot(steps, values, label=name, color=COLORS[name],
                linestyle=LINESTYLES[name], linewidth=LINEWIDTHS[name], alpha=0.85)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Validation PSNR — Ablation Study", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    # MSE
    ax = axes[1]
    for name in RUNS:
        if name not in data:
            continue
        steps, values = data[name]["val/mse"]
        if not steps:
            continue
        ax.plot(steps, values, label=name, color=COLORS[name],
                linestyle=LINESTYLES[name], linewidth=LINEWIDTHS[name], alpha=0.85)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("Validation MSE — Ablation Study", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "ablation_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def print_metrics_table(data):
    """Print best and final metrics, sorted by best PSNR descending."""
    rows = []
    for name in RUNS:
        if name not in data:
            continue
        steps_p, vals_p = data[name]["val/psnr"]
        steps_m, vals_m = data[name]["val/mse"]
        if not steps_p:
            continue
        best_idx = int(np.argmax(vals_p))
        best_mse_idx = int(np.argmin(vals_m))
        rows.append((name, vals_p[best_idx], steps_p[best_idx], vals_p[-1],
                      vals_m[best_mse_idx], vals_m[-1]))
    rows.sort(key=lambda r: r[1], reverse=True)

    print("\n" + "=" * 100)
    print(f"  {'Variant':<25} | {'Best PSNR':>10} @ {'Step':>6} | {'Final PSNR':>11} | {'Best MSE':>10} | {'Final MSE':>10}")
    print("  " + "-" * 94)
    for name, bp, bs, fp, bm, fm in rows:
        print(f"  {name:<25} | {bp:>8.2f}dB @ {bs:>5} | {fp:>9.2f}dB | {bm:>10.6f} | {fm:>10.6f}")
    print("=" * 100)

    # Degradation analysis
    print("\n  DEGRADATION (Best → Final PSNR):")
    for name, bp, bs, fp, bm, fm in rows:
        deg = fp - bp
        print(f"    {name:<25}: {deg:+.2f} dB")


def create_visual_comparison(sample_idx=0):
    """Side-by-side composite comparison — one row per variant, one column per milestone."""
    run_names = list(SAVE_DIRS.keys())
    n_runs = len(run_names)
    n_iters = len(MILESTONE_ITERS)

    cell_w, cell_h = 8, 2.2
    fig, axes = plt.subplots(
        n_runs, n_iters,
        figsize=(cell_w * n_iters, cell_h * n_runs + 1),
    )

    for row, name in enumerate(run_names):
        for col, it in enumerate(MILESTONE_ITERS):
            ax = axes[row, col]
            img_path = os.path.join(
                SAVE_DIRS[name], f"it{it}-test", "preview",
                f"composite_0_{sample_idx}_0.jpg"
            )
            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"iter {it}", fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(name, fontsize=13, fontweight="bold")

    fig.suptitle(
        f"Ablation Study — Qualitative Comparison (sample {sample_idx})",
        fontsize=15, fontweight="bold", y=1.0,
    )
    plt.subplots_adjust(wspace=0.05, hspace=0.08)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"visual_composite_s{sample_idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("[1/3] Extracting TB scalars...")
    data = extract_scalars()
    plot_training_curves(data)
    print_metrics_table(data)
    print("[3/3] Generating visual composites...")
    for s in range(10):
        create_visual_comparison(sample_idx=s)
    print(f"\nAll outputs saved to: {OUT_DIR}/")
