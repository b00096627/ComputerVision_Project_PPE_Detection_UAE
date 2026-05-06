import argparse
import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ────────────────────────────────────────────────────────────────────────────
# Augmentation pipeline
# ────────────────────────────────────────────────────────────────────────────

def _sandstorm_overlay(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Simulate a UAE khamseen / sandstorm by blending a warm sandy layer.
    The yellowish haze reduces visible contrast of hard-hats and vests,
    making detection harder — exactly the condition we want the model to learn.
    """
    # Sandy yellow-ochre colour to mimic the appearance of UAE dust in the air
    sand_color = np.array([205, 175, 120], dtype=np.uint8)
    overlay = np.full_like(image, sand_color)
    alpha   = np.random.uniform(0.25, 0.50)   # 25–50% opacity haze
    return cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0)


def _uae_sun_bleach(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Simulate UV colour bleaching from prolonged exposure to UAE noon sun.
    Hard-hat colours (especially yellow/orange) fade and lose saturation,
    making PPE classes harder to distinguish from the background.
    """
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Reduce saturation to simulate sun-bleached colours
    img_hsv[:, :, 1] *= np.random.uniform(0.35, 0.70)
    # Boost value (brightness) to simulate overlit conditions
    img_hsv[:, :, 2]  = np.clip(img_hsv[:, :, 2] * np.random.uniform(1.1, 1.4), 0, 255)
    return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def build_pipeline() -> A.Compose:
    """
    Build the UAE-tuned augmentation pipeline.

    Bounding boxes are passed in YOLO format (x_center, y_center, w, h)
    normalised to [0, 1].  Albumentations clips/removes boxes that fall
    outside the image after any spatial transform.

    Pipeline structure:
      1. Spatial   — camera angle, zoom, crop
      2. Standard colour — HSV, brightness/contrast, CLAHE
      3. UAE sun   — overexposure OR harsh shadow OR sun flare
      4. UAE environment — sandstorm haze OR colour bleach
      5. UAE heat shimmer — mild grid distortion
      6. UAE warm cast — RGB shift toward red/yellow
      7. Blur + noise
      8. Occlusion (CoarseDropout / Cutout)
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),

            # Simulate different camera tilt / zoom levels on scaffolding cams.
            A.Affine(
                translate_percent=0.05,
                scale=(0.8, 1.2),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            ),

            # Partial-view crop: camera only captures part of the site.
            A.RandomResizedCrop(size=(640, 640), scale=(0.70, 1.0), p=0.3),

            # HSV jitter covers normal day-to-day lighting variation.
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.4
            ),

            # CLAHE restores contrast in hazy / flat-lit images.
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),

            # Applying all three simultaneously would be physically unrealistic,
            # so OneOf ensures only one sun condition is applied per image.
            A.OneOf(
                [
                    # Noon overexposure: sun directly overhead washes out the image.
                    # brightness_limit upper bound of 0.9 = almost pure white.
                    A.RandomBrightnessContrast(
                        brightness_limit=(0.45, 0.90),
                        contrast_limit=(-0.25, 0.10),
                        p=1.0,
                    ),

                    # Harsh shadow: tall buildings / cranes cast very dark shadows.
                    # Workers partially in shadow are harder to detect.
                    A.RandomShadow(
                        shadow_roi=(0, 0.4, 1, 1),   # shadow falls in lower frame
                        num_shadows_limit=(1, 3),
                        shadow_dimension=6,
                        p=1.0,
                    ),

                    # Sun flare: rooftop cameras frequently face the sun directly,
                    # causing lens flare that occludes parts of the scene.
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),    # sun appears in upper half
                        src_radius=150,
                        p=1.0,
                    ),
                ],
                p=0.55,
            ),

            A.OneOf(
                [
                    # Sandstorm / khamseen: thick yellowish haze.
                    # Higher fog_coef than rain-country datasets — UAE dust is denser.
                    A.RandomFog(
                        fog_coef_range=(0.25, 0.55),
                        alpha_coef=0.15,
                        p=1.0,
                    ),

                    # Custom sandy overlay
                    A.Lambda(image=_sandstorm_overlay, p=1.0),

                    # UV colour bleaching
                    A.Lambda(image=_uae_sun_bleach, p=1.0),
                ],
                p=0.40,   # 40% chance any environmental effect is applied
            ),

            # Heat shimmer
            # Hot asphalt (50°C+) creates mirage-like wavy distortion at
            # ground level, modelled as a gentle grid warp.
            A.GridDistortion(num_steps=5, distort_limit=0.10, p=0.15),

            # UAE warm colour cast ─
            # Desert dust shifts the scene toward warm yellow-red.
            A.RGBShift(
                r_shift_limit=(0.02, 0.12),   # red:  +2 % … +12 % (warmer)
                g_shift_limit=(-0.03, 0.04),  # green: slight variation
                b_shift_limit=(-0.12, -0.02), # blue: −2 % … −12 % (warmer)
                p=0.35,
            ),

            # Blur / noise ─
            # Motion blur: camera shake on an unsecured pole mount.
            A.GaussianBlur(blur_limit=(3, 7), p=0.20),

            # Sensor noise: cameras running hot in 45°C ambient temperature.
            A.GaussNoise(std_range=(0.04, 0.20), p=0.20),

            # Light rain: rare but occurs in UAE winter (Dec–Feb).
            A.RandomRain(
                slant_range=(-10, 10),  
                drop_length=15,
                drop_width=1,
                blur_value=3,
                brightness_coefficient=0.9,
                rain_type="drizzle",
                p=0.08,
            ),

            # Occlusion
            # Forces the model to use context when hard-hats are partially
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=0.25,
            ),
        ],
        # Propagate YOLO-format bboxes through every spatial transform.
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.30,  # discard a box if <30% of its area remains
        ),
    )


# I/O helpers
# ────────────────────────────────────────────────────────────────────────────

def read_yolo_labels(label_path: Path) -> tuple[list, list]:
    """Read a YOLO label file. Returns (class_ids, bboxes_xywh_normalised)."""
    class_ids, bboxes = [], []
    if label_path.exists() and label_path.stat().st_size > 0:
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                cls = int(parts[0])
                box = [float(x) for x in parts[1:]]
                class_ids.append(cls)
                bboxes.append(box)
    return class_ids, bboxes


def write_yolo_labels(label_path: Path, class_ids: list, bboxes: list) -> None:
    """Write YOLO-format label file from class_ids and bboxes.

    Albumentations 2.x returns class labels as floats (e.g. 0.0 instead of 0).
    Casting to int here prevents YOLO from misreading the saved labels.
    """
    lines = [f"{int(c)} {' '.join(f'{v:.6f}' for v in b)}" for c, b in zip(class_ids, bboxes)]
    label_path.write_text("\n".join(lines))


def load_image_and_labels(img_path: Path, lbl_dir: Path):
    """Load an image (BGR to RGB) and its corresponding YOLO labels."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, [], []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lbl_path = lbl_dir / (img_path.stem + ".txt")
    class_ids, bboxes = read_yolo_labels(lbl_path)
    return img_rgb, class_ids, bboxes


# ────────────────────────────────────────────────────────────────────────────
# Demo mode: visualise augmentations
# ────────────────────────────────────────────────────────────────────────────

def demo_augmentations(
    img_paths: list,
    lbl_dir: Path,
    class_names: list,
    n_samples: int,
    output_dir: Path,
) -> None:
    """Show original vs augmented images side-by-side in a grid."""
    pipeline = build_pipeline()
    sampled  = random.sample(img_paths, min(n_samples, len(img_paths)))
    cols     = 4                           # original + 3 augmentations per image
    rows     = len(sampled)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    if rows == 1:
        axes = [axes]

    # Colour palette for bounding box labels
    cmap   = plt.get_cmap("tab10")
    colors = {i: tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(len(class_names))}

    def draw_boxes(image: np.ndarray, class_ids: list, bboxes: list) -> np.ndarray:
        """Draw YOLO-format boxes on a copy of the image."""
        h, w = image.shape[:2]
        out  = image.copy()
        for cls, (xc, yc, bw, bh) in zip(class_ids, bboxes):
            cls = int(cls)
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            col = colors.get(cls, (255, 0, 0))
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.putText(
                out, class_names[cls], (x1, max(y1 - 4, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA
            )
        return out

    for row_idx, img_path in enumerate(sampled):
        img, class_ids, bboxes = load_image_and_labels(img_path, lbl_dir)
        if img is None:
            continue

        imgs_to_show = [draw_boxes(img, class_ids, bboxes)]
        titles       = ["Original + GT"]

        for aug_i in range(cols - 1):
            try:
                aug = pipeline(image=img, bboxes=bboxes, class_labels=class_ids)
                aug_img = draw_boxes(aug["image"], aug["class_labels"], aug["bboxes"])
                imgs_to_show.append(aug_img)
                titles.append(f"Aug #{aug_i + 1}")
            except Exception as e:
                imgs_to_show.append(np.zeros_like(img))
                titles.append(f"Error: {e}")

        for col_idx, (im, title) in enumerate(zip(imgs_to_show, titles)):
            ax = axes[row_idx][col_idx]
            ax.imshow(im)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    plt.suptitle("Augmentation Pipeline Demo — PPE Dataset", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "augmentation_demo.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[OK] Augmentation demo saved: {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# Augment mode: write extra images to training split
# ────────────────────────────────────────────────────────────────────────────

def generate_augmented_images(
    img_paths: list,
    lbl_dir: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    n_samples: int,
) -> None:
    """
    Apply one augmentation per sampled image and save the result.
    Generated filenames are suffixed with '_aug<N>' to avoid collision.
    These extra images can be added to the train split to balance rare classes.
    """
    pipeline = build_pipeline()
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    sampled = random.choices(img_paths, k=n_samples)   # allow repeats
    saved   = 0

    for i, img_path in enumerate(tqdm(sampled, desc="Augmenting", unit="img")):
        img, class_ids, bboxes = load_image_and_labels(img_path, lbl_dir)
        if img is None or not bboxes:
            continue
        try:
            aug = pipeline(image=img, bboxes=bboxes, class_labels=class_ids)
        except Exception:
            continue
        if not aug["bboxes"]:
            continue   # all boxes were cropped out — skip

        stem     = img_path.stem + f"_aug{i:05d}"
        out_img  = out_img_dir / (stem + img_path.suffix)
        out_lbl  = out_lbl_dir / (stem + ".txt")

        aug_bgr = cv2.cvtColor(aug["image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_img), aug_bgr)
        write_yolo_labels(out_lbl, aug["class_labels"], aug["bboxes"])
        saved += 1

    print(f"[OK] Saved {saved}/{n_samples} augmented images: {out_img_dir}")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PPE augmentation pipeline demo and generator")
    parser.add_argument("--data",      default="configs/dataset.yaml")
    parser.add_argument("--mode",      choices=["demo", "augment"], default="demo",
                        help="demo: visualise pipeline | augment: write extra training images")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of images to sample/generate (default: 500 for augment, 12 for demo)")
    parser.add_argument("--output",    default="data/augmentation_demo",
                        help="Output directory for demo grid or augmented images")
    args = parser.parse_args()

    yaml_p = (PROJECT_ROOT / args.data).resolve()
    with open(yaml_p) as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg["path"])
    if not dataset_root.is_absolute():
        dataset_root = (yaml_p.parent / cfg["path"]).resolve()

    names    = cfg.get("names", {})
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    train_img_dir = dataset_root / "train" / "images"
    train_lbl_dir = dataset_root / "train" / "labels"

    img_ext  = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted(p for p in train_img_dir.glob("*") if p.suffix.lower() in img_ext)

    if not img_paths:
        sys.exit(f"No training images found in {train_img_dir}")

    print(f"Found {len(img_paths)} training images in {train_img_dir}")

    output_dir = (PROJECT_ROOT / args.output).resolve()

    if args.mode == "demo":
        demo_augmentations(img_paths, train_lbl_dir, names, args.n_samples, output_dir)
    else:
        out_img = output_dir / "images"
        out_lbl = output_dir / "labels"
        generate_augmented_images(img_paths, train_lbl_dir, out_img, out_lbl, args.n_samples)
        print("\nTo include augmented images in the training split, copy them over:")
        print(f"  # Linux/macOS:")
        print(f"  cp {out_img}/* {train_img_dir}/")
        print(f"  cp {out_lbl}/* {train_lbl_dir}/")
        print(f"  # Windows (PowerShell):")
        print(f"  Copy-Item '{out_img}\\*' '{train_img_dir}\\'")
        print(f"  Copy-Item '{out_lbl}\\*' '{train_lbl_dir}\\'")


if __name__ == "__main__":
    main()
