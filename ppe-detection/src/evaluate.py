#!/usr/bin/env python3
"""
evaluate.py — Evaluate trained YOLOv8 and RT-DETR models on the PPE test set.

Runs model.val() with split="test" for both models and exports per-class and
overall metrics to CSV so that compare.py can load them for visualisation.

Exported metrics per model:
  - Precision (P) per class
  - Recall    (R) per class
  - F1 score  per class   (computed as 2PR/(P+R))
  - AP@0.50   per class   (mAP50)
  - AP@0.50:0.95 per class (mAP50-95)
  - Overall mAP50 and mAP50-95 (added as a "Overall" row)

Output files:
  results/metrics/yolo_metrics.csv
  results/metrics/rtdetr_metrics.csv

Usage:
    python src/evaluate.py \\
        --yolo-weights   runs/yolo/weights/best.pt \\
        --rtdetr-weights runs/rtdetr/weights/best.pt \\
        --data           configs/dataset.yaml \\
        --imgsz 640 --device 0

    # Evaluate only one model
    python src/evaluate.py --yolo-weights runs/yolo/weights/best.pt --data configs/dataset.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR  = PROJECT_ROOT / "results" / "metrics"


# ────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on the PPE test split"
    )
    parser.add_argument("--yolo-weights",   default=None,
                        help="Path to trained YOLOv8 best.pt")
    parser.add_argument("--rtdetr-weights", default=None,
                        help="Path to trained RT-DETR best.pt")
    parser.add_argument("--data", default="configs/dataset.yaml",
                        help="Dataset yaml path (default: configs/dataset.yaml)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Validation batch size (default: 16)")
    parser.add_argument("--device", default="0",
                        help="CUDA device or 'cpu' (default: 0)")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold for mAP calc (default: 0.001)")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="IoU threshold for NMS (default: 0.6)")
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    weights_path: Path,
    data_yaml: Path,
    model_name: str,
    args: argparse.Namespace,
) -> pd.DataFrame | None:
    """
    Load the trained model, run validation on the test split, and return a
    DataFrame with per-class and overall metrics.

    We use split="test" (not "val") so the reported numbers reflect true
    generalisation — the test set was never seen during training or HP tuning.
    """
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics>=8.2")

    if not weights_path.exists():
        print(f"  [WARN] Weights not found: {weights_path}  — skipping {model_name}")
        return None

    print(f"\n{'─'*60}")
    print(f"  Evaluating: {model_name.upper()}  [{weights_path.name}]")
    print(f"{'─'*60}")

    if "rtdetr" in model_name.lower() or "rtdetr" in str(weights_path).lower():
        model = RTDETR(str(weights_path))
    else:
        model = YOLO(str(weights_path))

    # Run test-set validation
    # conf=0.001 is the standard setting for mAP evaluation — it ensures
    # the precision-recall curve covers the full recall range.
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        verbose=True,
        plots=True,
        project=str(PROJECT_ROOT / "runs"),
        name=f"{model_name}_eval",
        exist_ok=True,
    )

    # ── Extract metrics from the returned DetMetrics object
    box = metrics.box

    names_dict  = metrics.names
    class_names = [names_dict[k] for k in sorted(names_dict.keys())]
    nc          = len(class_names)

    ap_class_idx = np.asarray(getattr(box, "ap_class_index", []), dtype=int)

    raw_p    = np.asarray(getattr(box, "p",    []))
    raw_r    = np.asarray(getattr(box, "r",    []))
    raw_ap50 = np.asarray(getattr(box, "ap50", []))
    raw_ap   = np.asarray(getattr(box, "ap",   []))

    # Initialise full nc-length arrays with zeros (classes absent from the
    # test split stay at 0, which is the correct metric for them).
    p    = np.zeros(nc)
    r    = np.zeros(nc)
    ap50 = np.zeros(nc)
    ap   = np.zeros(nc)

    for i, cls_idx in enumerate(ap_class_idx):
        ci = int(cls_idx)
        if 0 <= ci < nc:
            if i < len(raw_p):    p[ci]    = raw_p[i]
            if i < len(raw_r):    r[ci]    = raw_r[i]
            if i < len(raw_ap50): ap50[ci] = raw_ap50[i]
            if i < len(raw_ap):   ap[ci]   = raw_ap[i]

    f1 = 2 * p * r / np.where((p + r) > 0, p + r, 1e-8)

    # ── Build per-class DataFrame ────────────────────────────────────────────
    df = pd.DataFrame({
        "class":     class_names,
        "precision": p.round(4),
        "recall":    r.round(4),
        "f1":        f1.round(4),
        "mAP50":     ap50.round(4),
        "mAP50_95":  ap.round(4),
    })

    # ── Append overall row (mean across classes / mAP values) ───────────────
    overall = pd.DataFrame([{
        "class":     "Overall",
        "precision": round(float(box.mp),    4),
        "recall":    round(float(box.mr),    4),
        "f1":        round(float(2 * box.mp * box.mr / max(box.mp + box.mr, 1e-8)), 4),
        "mAP50":     round(float(box.map50), 4),
        "mAP50_95":  round(float(box.map),   4),
    }])
    df = pd.concat([df, overall], ignore_index=True)

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"\n  {model_name.upper()} — Test Set Results")
    print(df.to_string(index=False))
    print(f"\n  Overall mAP50   : {box.map50:.4f}")
    print(f"  Overall mAP50-95: {box.map:.4f}")

    return df


# ────────────────────────────────────────────────────────────────────────────
# Save and report
# ────────────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, model_name: str) -> Path:
    """Save metrics DataFrame to results/metrics/<model_name>_metrics.csv."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / f"{model_name}_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  [OK] Metrics saved: {out_path}")
    return out_path


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    data_yaml = (PROJECT_ROOT / args.data).resolve()
    if not data_yaml.exists():
        sys.exit(f"Dataset yaml not found: {data_yaml}")

    if args.yolo_weights is None and args.rtdetr_weights is None:
        sys.exit("Provide at least one of --yolo-weights or --rtdetr-weights")

    saved_csvs = []

    # ── YOLOv8 ──────────────────────────────────────────────────────────────
    if args.yolo_weights:
        yolo_df = evaluate_model(
            Path(args.yolo_weights), data_yaml, "yolo", args
        )
        if yolo_df is not None:
            saved_csvs.append(save_csv(yolo_df, "yolo"))

    # ── RT-DETR ─────────────────────────────────────────────────────────────
    if args.rtdetr_weights:
        rtdetr_df = evaluate_model(
            Path(args.rtdetr_weights), data_yaml, "rtdetr", args
        )
        if rtdetr_df is not None:
            saved_csvs.append(save_csv(rtdetr_df, "rtdetr"))

    print(f"\n{'='*60}")
    print("  Evaluation complete.  CSVs saved:")
    for p in saved_csvs:
        print(f"    {p}")
    if len(saved_csvs) == 2:
        print(
            "\nNext step:\n"
            "  python src/compare.py \\\n"
            "      --yolo-metrics   results/metrics/yolo_metrics.csv \\\n"
            "      --rtdetr-metrics results/metrics/rtdetr_metrics.csv \\\n"
            "      --yolo-weights   runs/yolo/weights/best.pt \\\n"
            "      --rtdetr-weights runs/rtdetr/weights/best.pt\n"
        )


if __name__ == "__main__":
    main()
