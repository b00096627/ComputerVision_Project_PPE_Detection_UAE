#!/usr/bin/env python3
"""
visualize.py — Side-by-side prediction grid for YOLOv8 vs RT-DETR on test images.

Creates a figure with N rows × 3 columns:
  Column 0: Ground Truth (GT) boxes drawn from YOLO label files
  Column 1: YOLOv8 predictions
  Column 2: RT-DETR predictions

This qualitative analysis complements the quantitative metrics from evaluate.py
and is essential for the academic report — numbers alone don't reveal WHERE
each model succeeds or fails (e.g. crowded scenes, occluded workers).

Output: results/plots/prediction_grid.png

Usage:
    python src/visualize.py \\
        --yolo-weights   runs/yolo/weights/best.pt \\
        --rtdetr-weights runs/rtdetr/weights/best.pt \\
        --test-dir       data/construction_site_safety/test \\
        --output         results/plots/prediction_grid.png \\
        --n-images 6 --conf 0.25
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Class names and their corresponding BGR colours (for GT boxes)
CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask",
    "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest",
]

# Colours per class (BGR for OpenCV, converted to RGB for matplotlib)
CLASS_COLORS_BGR = [
    (0,   200, 0),    # Hardhat      — green
    (200, 200, 0),    # Mask         — yellow
    (0,   0,   255),  # NO-Hardhat   — red (violation)
    (0,   50,  255),  # NO-Mask      — orange-red (violation)
    (50,  0,   255),  # NO-Safety Vest — deep red (violation)
    (200, 200, 200),  # Person       — light grey
    (255, 130, 0),    # Safety Cone  — orange
    (0,   200, 130),  # Safety Vest  — teal
]

# Convert once to RGB fractions for matplotlib
CLASS_COLORS_RGB = [
    tuple(c / 255 for c in (bgr[2], bgr[1], bgr[0]))  # BGR → RGB
    for bgr in CLASS_COLORS_BGR
]


# ────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GT vs YOLO vs RT-DETR side-by-side prediction grid"
    )
    parser.add_argument("--yolo-weights",   required=True,
                        help="Path to trained YOLOv8 best.pt")
    parser.add_argument("--rtdetr-weights", required=True,
                        help="Path to trained RT-DETR best.pt")
    parser.add_argument("--test-dir",       required=True,
                        help="Test split root dir (must contain images/ and labels/)")
    parser.add_argument("--output",         default="results/plots/prediction_grid.png",
                        help="Output image path (default: results/plots/prediction_grid.png)")
    parser.add_argument("--n-images",       type=int, default=6,
                        help="Number of test images to visualise (default: 6)")
    parser.add_argument("--conf",           type=float, default=0.25,
                        help="Confidence threshold for displayed predictions (default: 0.25)")
    parser.add_argument("--device",         default="0",
                        help="CUDA device or 'cpu' (default: 0)")
    parser.add_argument("--seed",           type=int, default=42,
                        help="Random seed for image selection (default: 42)")
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ────────────────────────────────────────────────────────────────────────────

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB for matplotlib."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def draw_yolo_gt(img: np.ndarray, label_path: Path) -> np.ndarray:
    """
    Draw ground-truth boxes from a YOLO label file onto a copy of img.
    YOLO format: <class_id> <x_center> <y_center> <width> <height>  (normalised)
    """
    out = img.copy()
    h, w = out.shape[:2]

    if not label_path.exists() or label_path.stat().st_size == 0:
        return out

    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        color = CLASS_COLORS_BGR[cls % len(CLASS_COLORS_BGR)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        label   = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_predictions(img: np.ndarray, result, model_label: str) -> np.ndarray:
    """
    Draw prediction boxes from an Ultralytics result object.
    result.boxes.xyxy  — absolute pixel coords (x1, y1, x2, y2)
    result.boxes.cls   — class indices
    result.boxes.conf  — confidence scores
    """
    out = img.copy()
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        cv2.putText(out, f"{model_label}: no detections", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return out

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls  = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        color = CLASS_COLORS_BGR[cls % len(CLASS_COLORS_BGR)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        name    = result.names.get(cls, str(cls))
        label   = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Main grid builder
# ────────────────────────────────────────────────────────────────────────────

def build_grid(
    img_paths: list,
    lbl_dir: Path,
    yolo_model,
    rtdetr_model,
    n_images: int,
    conf: float,
    device: str,
    output_path: Path,
) -> None:
    """
    For each sampled image:
      - Read raw image and resize to a standard display size
      - Draw GT boxes (from .txt label file)
      - Run YOLO inference → draw predictions
      - Run RT-DETR inference → draw predictions
      - Place all three in the grid

    Using a consistent display size (640×480) ensures the figure columns
    align cleanly regardless of the original image aspect ratios.
    """
    random.shuffle(img_paths)       # already seeded before calling this
    sampled   = img_paths[:n_images]
    DISP_W, DISP_H = 640, 480       # display resolution per panel

    rows = len(sampled)
    cols = 3                        # GT | YOLO | RT-DETR

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.8))
    if rows == 1:
        axes = [axes]

    col_titles = ["Ground Truth", "YOLOv8m Predictions", "RT-DETR-L Predictions"]
    col_colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]   # green, blue, orange  # green, blue, orange

    # Column header row
    for c, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0][c].set_title(title, fontsize=13, fontweight="bold", color=color, pad=10)

    for row_idx, img_path in enumerate(sampled):
        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"  [WARN] Could not read {img_path}")
            for ax in axes[row_idx]:
                ax.axis("off")
            continue

        # Resize for consistent display
        raw_disp = cv2.resize(raw, (DISP_W, DISP_H))

        # ── Ground Truth ────────────────────────────────────────────────
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gt_img   = draw_yolo_gt(raw_disp, lbl_path)

        # ── YOLO predictions ────────────────────────────────────────────
        yolo_results = yolo_model.predict(
            str(img_path), conf=conf, device=device, verbose=False, imgsz=DISP_W
        )
        yolo_img = draw_predictions(raw_disp.copy(), yolo_results[0], "YOLO")

        # ── RT-DETR predictions ─────────────────────────────────────────
        rtdetr_results = rtdetr_model.predict(
            str(img_path), conf=conf, device=device, verbose=False, imgsz=DISP_W
        )
        rtdetr_img = draw_predictions(raw_disp.copy(), rtdetr_results[0], "RTDETR")

        # ── Lay out panels ──────────────────────────────────────────────
        panels = [gt_img, yolo_img, rtdetr_img]
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx][col_idx]
            ax.imshow(bgr_to_rgb(panel))
            ax.axis("off")
            # Row label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(
                    img_path.name, fontsize=7, rotation=0,
                    labelpad=60, va="center"
                )

    # ── Legend (class colours) ───────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=c, label=n)
        for n, c in zip(CLASS_NAMES, CLASS_COLORS_RGB)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=8,
        title="Class Legend",
        title_fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
        framealpha=0.9,
    )

    fig.suptitle(
        "PPE Detection — Qualitative Comparison\n"
        "Ground Truth  |  YOLOv8m  |  RT-DETR-L",
        fontsize=14, fontweight="bold", y=1.01
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])   # leave room for legend at bottom
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Prediction grid saved → {output_path}")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    try:
        from ultralytics import RTDETR, YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics>=8.2")

    # ── Paths ────────────────────────────────────────────────────────────────
    test_dir  = Path(args.test_dir)
    img_dir   = test_dir / "images"
    lbl_dir   = test_dir / "labels"
    out_path  = (PROJECT_ROOT / args.output).resolve()

    if not img_dir.exists():
        sys.exit(f"Test images directory not found: {img_dir}")

    img_ext   = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in img_ext)

    if not img_paths:
        sys.exit(f"No images found in {img_dir}")

    print(f"Found {len(img_paths)} test images")

    # ── Load models ──────────────────────────────────────────────────────────
    print(f"Loading YOLOv8 from  {args.yolo_weights} …")
    yolo_model = YOLO(args.yolo_weights)

    print(f"Loading RT-DETR from {args.rtdetr_weights} …")
    rtdetr_model = RTDETR(args.rtdetr_weights)

    # ── Seed and build grid ──────────────────────────────────────────────────
    random.seed(args.seed)
    build_grid(
        img_paths, lbl_dir,
        yolo_model, rtdetr_model,
        args.n_images, args.conf, args.device,
        out_path,
    )

    print("\nAll done! Open the grid image to compare predictions qualitatively.")
    print("Open the analysis notebook for quantitative results:")
    print("  jupyter notebook notebooks/analysis.ipynb")


if __name__ == "__main__":
    main()
