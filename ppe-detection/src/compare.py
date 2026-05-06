#!/usr/bin/env python3
"""
compare.py — Load evaluation CSVs, generate comparison plots, and benchmark FPS.

Produces results/plots/comparison.png with four subplots:
  1. Overall mAP50     bar chart (YOLO vs RT-DETR)
  2. Overall mAP50-95  bar chart
  3. Inference FPS     bar chart (from live benchmark on test images)
  4. Per-class F1      grouped bar chart

Also prints a training-time comparison if epoch_times.json files exist in the
run directories.

Usage:
    python src/compare.py \\
        --yolo-metrics   results/metrics/yolo_metrics.csv \\
        --rtdetr-metrics results/metrics/rtdetr_metrics.csv \\
        --yolo-weights   runs/yolo/weights/best.pt \\
        --rtdetr-weights runs/rtdetr/weights/best.pt \\
        --test-images    data/construction_site_safety/test/images \\
        --device 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR    = PROJECT_ROOT / "results" / "plots"

# Colour scheme: consistent across all plots so readers can follow YOLO vs RTDETR
YOLO_COLOR   = "#4C72B0"   # blue
RTDETR_COLOR = "#DD8452"   # orange


# ────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YOLO vs RT-DETR comparison plots and FPS benchmark"
    )
    parser.add_argument("--yolo-metrics",   required=True,
                        help="Path to results/metrics/yolo_metrics.csv")
    parser.add_argument("--rtdetr-metrics", required=True,
                        help="Path to results/metrics/rtdetr_metrics.csv")
    parser.add_argument("--yolo-weights",   default=None,
                        help="YOLOv8 best.pt for FPS benchmark (optional)")
    parser.add_argument("--rtdetr-weights", default=None,
                        help="RT-DETR best.pt for FPS benchmark (optional)")
    parser.add_argument("--test-images",    default=None,
                        help="Directory of test images for FPS benchmark")
    parser.add_argument("--device",         default="0",
                        help="CUDA device or 'cpu' (default: 0)")
    parser.add_argument("--n-benchmark",    type=int, default=100,
                        help="Number of test images to use for FPS benchmark (default: 100)")
    parser.add_argument("--warmup",         type=int, default=10,
                        help="Warmup inferences before timing (default: 10)")
    parser.add_argument("--output",         default="results/plots/comparison.png",
                        help="Output path for the comparison figure")
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# FPS benchmark
# ────────────────────────────────────────────────────────────────────────────

def benchmark_fps(
    weights_path: Path,
    img_paths: list,
    model_name: str,
    device: str,
    n_benchmark: int,
    warmup: int,
) -> float:
    """
    Measure inference throughput (frames per second) for one model.

    Why FPS matters:
      A safety monitoring system must process camera feeds in real time.
      Even if RT-DETR has higher mAP, a 2× speed penalty may make it
      impractical for a live multi-camera deployment.

    Methodology:
      1. Warmup N inferences (GPU pipeline initialisation, kernel caching).
      2. Time M inferences in a tight loop.
      3. FPS = M / elapsed_seconds.
    """
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError:
        print("  [WARN] ultralytics not installed — FPS benchmark skipped")
        return 0.0

    if "rtdetr" in model_name.lower():
        model = RTDETR(str(weights_path))
    else:
        model = YOLO(str(weights_path))

    sample = img_paths[:n_benchmark]

    print(f"\n  [FPS] Warming up {model_name.upper()} ({warmup} inferences) …")
    for p in sample[:warmup]:
        model.predict(str(p), device=device, verbose=False)

    print(f"  [FPS] Benchmarking {len(sample)} images …")
    t0 = time.perf_counter()
    for p in sample:
        model.predict(str(p), device=device, verbose=False)
    elapsed = time.perf_counter() - t0
    fps = len(sample) / elapsed

    print(f"  [FPS] {model_name.upper()}: {fps:.1f} FPS  ({elapsed:.2f}s for {len(sample)} images)")
    return round(fps, 1)


# ────────────────────────────────────────────────────────────────────────────
# Training time comparison
# ────────────────────────────────────────────────────────────────────────────

def load_epoch_times(model_name: str) -> dict | None:
    """Load epoch_times.json saved by train.py, if it exists."""
    log_path = PROJECT_ROOT / "runs" / model_name / "epoch_times.json"
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return None


def print_training_comparison() -> None:
    """Print a side-by-side training efficiency summary."""
    yolo_log   = load_epoch_times("yolo")
    rtdetr_log = load_epoch_times("rtdetr")

    if not yolo_log and not rtdetr_log:
        return

    print(f"\n{'─'*55}")
    print("  Training Efficiency Comparison")
    print(f"{'─'*55}")

    def fmt(log, name):
        if log is None:
            print(f"  {name}: timing log not found")
            return
        print(f"  {name}")
        print(f"    Total time   : {log['total_time_min']:.1f} min")
        print(f"    Epochs       : {log['epochs_completed']}")
        print(f"    Avg epoch    : {log['avg_epoch_sec']:.1f}s")

    fmt(yolo_log,   "YOLOv8m")
    fmt(rtdetr_log, "RT-DETR-L")


# ────────────────────────────────────────────────────────────────────────────
# Plot generation
# ────────────────────────────────────────────────────────────────────────────

def load_metrics(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a metrics CSV.  Returns (per_class_df, overall_series).
    The 'Overall' row is separated out so the per-class plot is clean.
    """
    df = pd.read_csv(csv_path)
    overall = df[df["class"] == "Overall"].iloc[0]
    per_cls = df[df["class"] != "Overall"].reset_index(drop=True)
    return per_cls, overall


def plot_bar(ax, values: list, labels: list, colors: list, title: str, ylabel: str) -> None:
    """Utility: horizontal bar chart for a scalar metric comparison."""
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.5, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )


def generate_comparison_figure(
    yolo_cls: pd.DataFrame,
    rtdetr_cls: pd.DataFrame,
    yolo_overall: pd.Series,
    rtdetr_overall: pd.Series,
    yolo_fps: float,
    rtdetr_fps: float,
    output_path: Path,
) -> None:
    """
    Create the 4-subplot comparison figure:
      [0] mAP50 overall bar
      [1] mAP50-95 overall bar
      [2] FPS bar (inference speed)
      [3] Per-class F1 grouped bar
    """
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    model_labels = ["YOLOv8m", "RT-DETR-L"]
    colors       = [YOLO_COLOR, RTDETR_COLOR]

    # ── [0] mAP50 ────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    plot_bar(
        ax0,
        values=[float(yolo_overall["mAP50"]), float(rtdetr_overall["mAP50"])],
        labels=model_labels, colors=colors,
        title="Overall mAP@0.50",
        ylabel="mAP50",
    )

    # ── [1] mAP50-95 ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    plot_bar(
        ax1,
        values=[float(yolo_overall["mAP50_95"]), float(rtdetr_overall["mAP50_95"])],
        labels=model_labels, colors=colors,
        title="Overall mAP@0.50:0.95",
        ylabel="mAP50-95",
    )

    # ── [2] FPS ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    fps_vals = [yolo_fps, rtdetr_fps]
    if max(fps_vals) > 0:
        plot_bar(
            ax2,
            values=fps_vals,
            labels=model_labels, colors=colors,
            title="Inference Speed (FPS ↑ is better)",
            ylabel="Frames per Second",
        )
    else:
        ax2.text(0.5, 0.5, "FPS benchmark\nnot available\n(provide --yolo-weights\nand --rtdetr-weights)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)
        ax2.set_title("Inference Speed (FPS)", fontsize=12, fontweight="bold")
        ax2.axis("off")

    # ── [3] Per-class F1 grouped bar ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    classes = yolo_cls["class"].tolist()

    yolo_f1   = yolo_cls.set_index("class")["f1"].reindex(classes).fillna(0).values
    rtdetr_f1 = rtdetr_cls.set_index("class")["f1"].reindex(classes).fillna(0).values

    x     = np.arange(len(classes))
    width = 0.35
    ax3.bar(x - width / 2, yolo_f1,   width, label="YOLOv8m",   color=YOLO_COLOR,   alpha=0.9)
    ax3.bar(x + width / 2, rtdetr_f1, width, label="RT-DETR-L", color=RTDETR_COLOR, alpha=0.9)

    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax3.set_ylabel("F1 Score", fontsize=10)
    ax3.set_title("Per-Class F1 Score Comparison", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 1.15)
    ax3.legend(fontsize=10)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bars
    for xi, (y_val, r_val) in enumerate(zip(yolo_f1, rtdetr_f1)):
        ax3.text(xi - width / 2, y_val + 0.01, f"{y_val:.2f}", ha="center", fontsize=7)
        ax3.text(xi + width / 2, r_val + 0.01, f"{r_val:.2f}", ha="center", fontsize=7)

    # ── Overall figure title ───────────────────────────────────────────────
    fig.suptitle(
        "YOLOv8m vs RT-DETR-L — PPE Detection Comparison\n"
        "(Construction Site Safety Dataset, 8 Classes)",
        fontsize=15, fontweight="bold", y=0.98
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Comparison figure saved → {output_path}")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    yolo_csv   = Path(args.yolo_metrics)
    rtdetr_csv = Path(args.rtdetr_metrics)

    for p in (yolo_csv, rtdetr_csv):
        if not p.exists():
            sys.exit(f"Metrics CSV not found: {p}\nRun src/evaluate.py first.")

    yolo_cls,   yolo_overall   = load_metrics(yolo_csv)
    rtdetr_cls, rtdetr_overall = load_metrics(rtdetr_csv)

    # ── FPS benchmark ───────────────────────────────────────────────────────
    yolo_fps   = 0.0
    rtdetr_fps = 0.0

    if args.yolo_weights and args.rtdetr_weights:
        # Collect test images for the benchmark
        if args.test_images:
            test_dir = Path(args.test_images)
        else:
            # Try to infer from yaml path pattern
            test_dir = PROJECT_ROOT / "data" / "construction_site_safety" / "test" / "images"

        img_ext   = {".jpg", ".jpeg", ".png", ".bmp"}
        img_paths = sorted(p for p in test_dir.glob("*") if p.suffix.lower() in img_ext)

        if not img_paths:
            print(f"  [WARN] No test images found in {test_dir} — FPS benchmark skipped")
        else:
            yolo_fps = benchmark_fps(
                Path(args.yolo_weights), img_paths, "yolo",
                args.device, args.n_benchmark, args.warmup
            )
            rtdetr_fps = benchmark_fps(
                Path(args.rtdetr_weights), img_paths, "rtdetr",
                args.device, args.n_benchmark, args.warmup
            )
    else:
        print("\n  [INFO] --yolo-weights and --rtdetr-weights not provided — FPS subplot will be empty.")

    # ── Print training time comparison ──────────────────────────────────────
    print_training_comparison()

    # ── Print accuracy summary table ────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  Accuracy Summary")
    print(f"{'─'*55}")
    print(f"  {'Metric':<15} {'YOLOv8m':>10} {'RT-DETR-L':>12}")
    print(f"  {'-'*40}")
    for col in ("mAP50", "mAP50_95", "precision", "recall", "f1"):
        y_val = float(yolo_overall[col])
        r_val = float(rtdetr_overall[col])
        winner = "← YOLO" if y_val > r_val else "← RTDETR" if r_val > y_val else ""
        print(f"  {col:<15} {y_val:>10.4f} {r_val:>12.4f}  {winner}")
    print(f"  {'FPS':<15} {yolo_fps:>10.1f} {rtdetr_fps:>12.1f}")

    # ── Generate figure ─────────────────────────────────────────────────────
    output_path = (PROJECT_ROOT / args.output).resolve()
    generate_comparison_figure(
        yolo_cls, rtdetr_cls,
        yolo_overall, rtdetr_overall,
        yolo_fps, rtdetr_fps,
        output_path,
    )

    print("\nNext step:")
    print("  python src/visualize.py \\")
    print("      --yolo-weights   runs/yolo/weights/best.pt \\")
    print("      --rtdetr-weights runs/rtdetr/weights/best.pt \\")
    print("      --test-dir data/construction_site_safety/test")


if __name__ == "__main__":
    main()
