#!/usr/bin/env python3
"""
train.py — Unified training script for YOLOv8 and RT-DETR on the PPE dataset.

Both models are trained via the Ultralytics API which provides a consistent
interface, shared augmentation pipeline, and identical evaluation metrics —
making the comparison fair and reproducible.

Architecture notes:
  YOLOv8m  — CNN-based one-stage detector with decoupled head; ~25M params.
  RT-DETR-L — Transformer-based detector (encoder-decoder with IoU-aware
              query selection); ~32M params.  Requires a larger GPU memory
              budget, hence the smaller default batch size.

Usage:
    # Train YOLOv8 medium
    python src/train.py --model yolo --epochs 50 --imgsz 640 --batch 16 --device 0

    # Train RT-DETR large
    python src/train.py --model rtdetr --epochs 50 --imgsz 640 --batch 4 --device 0

    # CPU (slow — for smoke-testing only)
    python src/train.py --model yolo --epochs 2 --batch 4 --device cpu
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 or RT-DETR on the PPE dataset"
    )
    parser.add_argument(
        "--model", choices=["yolo", "rtdetr"], required=True,
        help="Which model architecture to train"
    )
    parser.add_argument(
        "--data", default="configs/dataset.yaml",
        help="Path to dataset yaml (default: configs/dataset.yaml)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size in pixels (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Batch size (default: 16 for YOLO, 4 for RT-DETR)"
    )
    parser.add_argument(
        "--device", default="0",
        help="CUDA device index or 'cpu' (default: 0)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="DataLoader worker threads (default: 4)"
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Early-stopping patience in epochs (default: 20)"
    )
    parser.add_argument(
        "--pretrained", action="store_true", default=True,
        help="Start from COCO-pretrained weights (default: True)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last checkpoint if it exists"
    )
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Per-epoch timing callback
# ────────────────────────────────────────────────────────────────────────────

def make_timing_callbacks(epoch_log: list) -> dict:
    """
    Return Ultralytics-compatible callbacks that record the wall-clock time
    of every training epoch.  This data lets us compare training efficiency:
    RT-DETR's attention mechanism is slower per epoch than YOLO's CNN layers.

    IMPORTANT — callbacks must NEVER raise exceptions.  Ultralytics 8.4.x
    silently swallows callback errors but aborts the training loop when one
    occurs, causing the process to exit without any traceback.  Every access
    to trainer state is therefore wrapped in try/except.
    """
    _start = [0.0]   # mutable container so closures can mutate it

    def on_train_epoch_start(trainer):
        _start[0] = time.perf_counter()

    def on_train_epoch_end(trainer):
        elapsed = time.perf_counter() - _start[0]

        map50 = map50_95 = 0.0
        try:
            m = trainer.metrics
            if isinstance(m, dict):
                map50    = float(m.get("metrics/mAP50(B)", 0.0))
                map50_95 = float(m.get("metrics/mAP50-95(B)", 0.0))
            elif hasattr(m, "results_dict"):
                map50    = float(m.results_dict.get("metrics/mAP50(B)", 0.0))
                map50_95 = float(m.results_dict.get("metrics/mAP50-95(B)", 0.0))
            elif hasattr(m, "box"):
                map50    = float(getattr(m.box, "map50", 0.0))
                map50_95 = float(getattr(m.box, "map",   0.0))
        except Exception:
            pass   # metrics not yet available (e.g. first epoch before val)

        train_loss = None
        try:
            if hasattr(trainer, "loss") and trainer.loss is not None:
                train_loss = float(trainer.loss.item())
        except Exception:
            pass

        entry = {
            "epoch":      trainer.epoch + 1,
            "time_sec":   round(elapsed, 3),
            "time_min":   round(elapsed / 60, 3),
            "map50":      map50,
            "map50_95":   map50_95,
            "train_loss": train_loss,
        }
        epoch_log.append(entry)
        print(
            f"  [Timing] Epoch {entry['epoch']:3d}/{trainer.epochs}"
            f" | {elapsed:6.1f}s | mAP50={entry['map50']:.4f}"
        )

    return {
        "on_train_epoch_start": on_train_epoch_start,
        "on_train_epoch_end":   on_train_epoch_end,
    }


# ────────────────────────────────────────────────────────────────────────────
# Model factory
# ────────────────────────────────────────────────────────────────────────────

def load_model(model_key: str, pretrained: bool):
    """
    Load the correct Ultralytics model class.

    - 'yolo'   → YOLOv8m (medium variant; good accuracy/speed tradeoff)
    - 'rtdetr' → RT-DETR-L (large variant; Ultralytics' recommended size)

    Both start from COCO-pretrained weights which are auto-downloaded on
    first run.  Fine-tuning from COCO is much faster than training from
    scratch and gives better final accuracy on small custom datasets.
    """
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics>=8.2")

    if model_key == "yolo":
        weight = "yolov8m.pt" if pretrained else "yolov8m.yaml"
        print(f"[Model] Loading YOLOv8m from '{weight}' …")
        return YOLO(weight), "yolo"
    else:
        weight = "rtdetr-l.pt" if pretrained else "rtdetr-l.yaml"
        print(f"[Model] Loading RT-DETR-L from '{weight}' …")
        return RTDETR(weight), "rtdetr"


# ────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── Resolve data yaml to absolute path ──────────────────────────────────
    data_yaml = (PROJECT_ROOT / args.data).resolve()
    if not data_yaml.exists():
        sys.exit(f"Dataset yaml not found: {data_yaml}\nRun data/download_dataset.py first.")

    # ── Default batch sizes (RT-DETR uses more GPU memory) ─────────────────
    batch = args.batch or (16 if args.model == "yolo" else 4)

    # ── Load model ──────────────────────────────────────────────────────────
    model, run_name = load_model(args.model, args.pretrained)

    # ── Register epoch-timing callbacks ─────────────────────────────────────
    epoch_log: list = []
    callbacks = make_timing_callbacks(epoch_log)
    for event, fn in callbacks.items():
        model.add_callback(event, fn)

    # ── CPU-specific adjustments ────────────────────────────────────────────
    is_cpu = str(args.device).lower() == "cpu"

    use_amp = not is_cpu

    effective_workers = 0 if is_cpu else args.workers

    print(f"\n{'='*65}")
    print(f"  Training  : {run_name.upper()}")
    print(f"  Data      : {data_yaml}")
    print(f"  Epochs    : {args.epochs}   Imgsz: {args.imgsz}   Batch: {batch}")
    print(f"  Device    : {args.device}   AMP: {use_amp}   Workers: {effective_workers}")
    print(f"{'='*65}")

    if is_cpu:
        print(f"\n  [WARN] CPU training is very slow (~30-90 min per epoch).")
        print(f"  Recommended: use Google Colab (free T4 GPU) for 50 epochs.")
        print(f"  For a quick CPU smoke-test use: --epochs 2 --batch 4\n")

    wall_start = time.perf_counter()

    # ── Train ────────────────────────────────────────────────────────────────
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=effective_workers,
        patience=args.patience,
        resume=args.resume,
        amp=use_amp,
        project=str(PROJECT_ROOT / "runs"),
        name=run_name,
        exist_ok=True,
        # ── Ultralytics built-in augmentation ─────────────────────────────
        mosaic=1.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        # ── Optimiser ─────────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=3,
        # ── Misc ──────────────────────────────────────────────────────────
        plots=True,
        save=True,
        save_period=-1,
        val=True,
        verbose=True,
        seed=42,
    )

    total_time = time.perf_counter() - wall_start

    # ── Save timing log ─────────────────────────────────────────────────────
    run_dir  = PROJECT_ROOT / "runs" / run_name
    log_path = run_dir / "epoch_times.json"
    summary  = {
        "model":            run_name,
        "total_time_sec":   round(total_time, 2),
        "total_time_min":   round(total_time / 60, 2),
        "epochs_completed": len(epoch_log),
        "avg_epoch_sec":    round(sum(e["time_sec"] for e in epoch_log) / max(len(epoch_log), 1), 2),
        "epochs":           epoch_log,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2))

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Training complete — {run_name.upper()}")
    print(f"  Total wall time : {total_time/60:.1f} min")
    print(f"  Avg epoch time  : {summary['avg_epoch_sec']:.1f}s")
    print(f"  Best weights    : {run_dir}/weights/best.pt")
    print(f"  Epoch timing    : {log_path}")
    print(f"{'='*65}")
    print(
        "\nNext step:\n"
        "  python src/evaluate.py \\\n"
        f"      --{'yolo' if run_name == 'yolo' else 'rtdetr'}-weights "
        f"{run_dir}/weights/best.pt \\\n"
        "      --data configs/dataset.yaml\n"
    )


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
