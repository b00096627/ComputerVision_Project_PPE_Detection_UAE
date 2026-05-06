#!/usr/bin/env python3
"""
preprocess.py — Dataset validation, class-distribution analysis, and bad-image removal.

Why this step matters:
  Corrupted images or labels cause silent crashes mid-training.
  Class imbalance (e.g. 10× more "Person" than "NO-Hardhat") needs to be
  understood before interpreting per-class metrics.

Usage:
    python src/preprocess.py --data configs/dataset.yaml
    python src/preprocess.py --data configs/dataset.yaml --remove-bad
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PPE dataset and analyse class distribution"
    )
    parser.add_argument(
        "--data", default="configs/dataset.yaml",
        help="Path to Ultralytics dataset yaml (default: configs/dataset.yaml)"
    )
    parser.add_argument(
        "--remove-bad", action="store_true",
        help="If set, delete corrupted/unreadable images AND their label files"
    )
    parser.add_argument(
        "--min-dim", type=int, default=32,
        help="Minimum image dimension (px) to be considered valid (default: 32)"
    )
    parser.add_argument(
        "--output", default="results",
        help="Directory for saving the class-distribution plot (default: results)"
    )
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def load_dataset_config(yaml_path: Path) -> dict:
    """Parse dataset yaml and resolve the dataset root to an absolute path."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    raw_path = cfg.get("path", "")
    dataset_root = Path(raw_path)
    if not dataset_root.is_absolute():
        # Resolve relative to the yaml file's parent directory
        dataset_root = (yaml_path.parent / raw_path).resolve()

    cfg["_root"] = dataset_root
    cfg["_names"] = cfg.get("names", {})
    # Normalise names to a list regardless of whether yaml used list or dict
    if isinstance(cfg["_names"], dict):
        cfg["_names"] = [cfg["_names"][k] for k in sorted(cfg["_names"])]
    return cfg


def get_split_paths(cfg: dict, split: str):
    """Return (images_dir, labels_dir) for a given split."""
    root      = cfg["_root"]
    split_key = "val" if split == "valid" else split
    rel_img   = cfg.get(split_key, f"{split}/images")
    img_dir   = root / rel_img
    lbl_dir   = img_dir.parent.parent / split / "labels" if "images" in rel_img else img_dir.parent / "labels"
    # More robust: always derive labels dir from images dir
    lbl_dir = Path(str(img_dir).replace("images", "labels"))
    return img_dir, lbl_dir


def validate_image(img_path: Path, min_dim: int) -> tuple[bool, str]:
    """
    Try to open the image with PIL (catches truncated files) and then with
    OpenCV (catches some additional codec issues).
    Returns (is_valid, reason).
    """
    try:
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            w, h = img.size
        if min(w, h) < min_dim:
            return False, f"too small ({w}×{h})"
    except Exception as e:
        return False, str(e)

    # Double-check with OpenCV (catches some palette-mode issues PIL misses)
    frame = cv2.imread(str(img_path))
    if frame is None:
        return False, "OpenCV could not decode"
    return True, "ok"


def count_class_instances(label_dir: Path, nc: int) -> np.ndarray:
    """
    Read all YOLO label files in label_dir and count instances per class.
    Each label line: <class_id> <x_c> <y_c> <w> <h>
    Returns array of shape (nc,).
    """
    counts = np.zeros(nc, dtype=np.int64)
    for lbl_file in label_dir.glob("*.txt"):
        if lbl_file.stat().st_size == 0:
            continue
        for line in lbl_file.read_text().strip().splitlines():
            parts = line.split()
            if parts:
                cls_id = int(parts[0])
                if 0 <= cls_id < nc:
                    counts[cls_id] += 1
    return counts


# ────────────────────────────────────────────────────────────────────────────
# Main validation loop
# ────────────────────────────────────────────────────────────────────────────

def validate_split(
    img_dir: Path,
    lbl_dir: Path,
    split: str,
    min_dim: int,
    remove_bad: bool
) -> dict:
    """Validate all images in one split. Returns summary dict."""
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in img_extensions)

    n_total   = len(images)
    bad_files = []
    missing_labels = []

    print(f"\n{'─'*60}")
    print(f"  Validating split: {split}  ({n_total} images)")
    print(f"{'─'*60}")

    for img_path in tqdm(images, desc=f"  {split}", unit="img"):
        # Check image integrity
        valid, reason = validate_image(img_path, min_dim)
        if not valid:
            bad_files.append((img_path, reason))
            if remove_bad:
                img_path.unlink(missing_ok=True)
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                lbl_path.unlink(missing_ok=True)
            continue

        # Check corresponding label file exists
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            missing_labels.append(img_path)

    print(f"  Total images   : {n_total}")
    print(f"  Bad / corrupt  : {len(bad_files)}")
    print(f"  Missing labels : {len(missing_labels)}")

    if bad_files:
        print(f"\n  {'Removed' if remove_bad else 'Bad'} files:")
        for p, reason in bad_files[:10]:     # show at most 10
            print(f"    {p.name} → {reason}")
        if len(bad_files) > 10:
            print(f"    … and {len(bad_files)-10} more")

    return {
        "split": split,
        "total": n_total,
        "bad": len(bad_files),
        "missing_labels": len(missing_labels),
    }


# ────────────────────────────────────────────────────────────────────────────
# Class distribution plot
# ────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    counts_per_split: dict,   # {"train": np.ndarray, "valid": np.ndarray, "test": np.ndarray}
    class_names: list,
    output_dir: Path
) -> None:
    """
    Grouped bar chart showing instance counts per class for each split.
    Understanding the distribution is critical: heavily imbalanced classes
    (e.g. Person >> NO-Hardhat) can cause the model to under-detect rare
    violations.
    """
    splits = list(counts_per_split.keys())
    nc     = len(class_names)
    x      = np.arange(nc)
    width  = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (split, color) in enumerate(zip(splits, colors)):
        counts = counts_per_split[split]
        bars = ax.bar(x + i * width, counts, width, label=split.capitalize(), color=color, alpha=0.85)
        # Label the top of each bar with the count for readability in the report
        for bar, val in zip(bars, counts):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01,
                    str(val), ha="center", va="bottom", fontsize=7, rotation=45
                )

    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Instance Count", fontsize=12)
    ax.set_title("PPE Dataset — Class Instance Distribution per Split", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "class_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n✓ Class distribution plot saved → {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    yaml_p = (PROJECT_ROOT / args.data).resolve()

    if not yaml_p.exists():
        sys.exit(f"Dataset yaml not found: {yaml_p}\nRun data/download_dataset.py first.")

    cfg  = load_dataset_config(yaml_p)
    nc   = cfg.get("nc", 8)
    names = cfg["_names"]

    # ── Validate each split ─────────────────────────────────────────────────
    counts_per_split = {}
    for split in ("train", "valid", "test"):
        img_dir, lbl_dir = get_split_paths(cfg, split)
        if not img_dir.exists():
            print(f"  [SKIP] {split} images dir not found: {img_dir}")
            continue
        validate_split(img_dir, lbl_dir, split, args.min_dim, args.remove_bad)
        counts_per_split[split] = count_class_instances(lbl_dir, nc)

    # ── Print class distribution table ─────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Class Distribution Summary")
    print(f"{'─'*70}")
    header = f"  {'Class':<22}" + "".join(f"  {s.capitalize():>8}" for s in counts_per_split)
    print(header)
    print(f"  {'-'*65}")
    for i, name in enumerate(names):
        row = f"  {name:<22}"
        for counts in counts_per_split.values():
            row += f"  {counts[i]:>8,}"
        print(row)

    # ── Plot ────────────────────────────────────────────────────────────────
    if counts_per_split:
        output_dir = (PROJECT_ROOT / args.output / "plots").resolve()
        plot_class_distribution(counts_per_split, names, output_dir)

    print("\nPreprocessing complete. Next step:")
    print("  python src/train.py --model yolo   --epochs 50 --imgsz 640 --batch 16")
    print("  python src/train.py --model rtdetr --epochs 50 --imgsz 640 --batch 4")


if __name__ == "__main__":
    main()
