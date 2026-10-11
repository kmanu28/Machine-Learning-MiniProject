"""
Dataset Loader, Augmentation, and Subset Sampling for Airbus Ship Detection.
Supports:
1. Loading custom/Kaggle dataset with configurable subset limits (e.g. 500-2000 images).
2. Synthetic sample generator for immediate testing without 30GB downloads.
3. Data augmentation (rotations, flips, zoom, noise) as described in CS229 paper.
4. Preprocessing & feature caching for fast training.
"""

import os
import sys
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Tuple, List, Optional
import json

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
except ImportError:
    cv2 = None

from src.features import extract_features


def augment_image(img_rgb: np.ndarray) -> np.ndarray:
    """
    Applies random augmentations:
    - Random rotation (0, 90, 180, 270 degrees or slight angle)
    - Horizontal / Vertical flips
    - Slight brightness / contrast variation
    """
    augmented = img_rgb.copy()

    # Random flip
    if np.random.rand() > 0.5:
        augmented = np.fliplr(augmented)
    if np.random.rand() > 0.5:
        augmented = np.flipud(augmented)

    # Random 90-degree rotations
    k = np.random.randint(0, 4)
    if k > 0:
        augmented = np.rot90(augmented, k)

    # Random brightness shift (+-15%)
    shift = np.random.uniform(0.85, 1.15)
    augmented = np.clip(augmented * shift, 0, 255).astype(np.uint8)

    return augmented


def generate_synthetic_samples(
    output_dir: str = "data/sample_dataset",
    num_ship_samples: int = 150,
    num_no_ship_samples: int = 250,
    image_size: Tuple[int, int] = (256, 256)
) -> dict:
    """
    Generates realistic synthetic satellite imagery patches for testing & demonstration:
    - No-Ship: Open blue/green water with waves, coastlines, clouds/fog, wharfs.
    - Ship: Vessels of various sizes, orientations, wakes, and maritime backgrounds.
    """
    os.makedirs(f"{output_dir}/ships", exist_ok=True)
    os.makedirs(f"{output_dir}/no_ships", exist_ok=True)

    print(f"Generating synthetic dataset in '{output_dir}' ({num_ship_samples} ships, {num_no_ship_samples} no-ships)...")

    # 1. Generate No-Ship Images
    for i in range(num_no_ship_samples):
        # Base ocean color (dark blue, teal, or greenish ocean)
        base_color = (
            np.random.randint(15, 45),
            np.random.randint(45, 95),
            np.random.randint(70, 140)
        )
        img = Image.new("RGB", image_size, color=base_color)
        draw = ImageDraw.Draw(img)

        # Add wave textures
        for _ in range(np.random.randint(10, 30)):
            x1 = np.random.randint(0, image_size[0])
            y1 = np.random.randint(0, image_size[1])
            x2 = x1 + np.random.randint(15, 60)
            y2 = y1 + np.random.randint(-5, 5)
            wave_col = (
                min(255, base_color[0] + np.random.randint(10, 30)),
                min(255, base_color[1] + np.random.randint(15, 40)),
                min(255, base_color[2] + np.random.randint(20, 50))
            )
            draw.line([(x1, y1), (x2, y2)], fill=wave_col, width=np.random.randint(1, 3))

        # 30% chance of coastline / wharf / land patch
        if np.random.rand() > 0.7:
            land_color = (np.random.randint(80, 130), np.random.randint(90, 140), np.random.randint(60, 90))
            poly = [
                (0, 0),
                (np.random.randint(40, 100), 0),
                (0, np.random.randint(40, 100))
            ]
            draw.polygon(poly, fill=land_color)

        # 25% chance of cloud / fog patch
        if np.random.rand() > 0.75:
            cloud_color = (np.random.randint(210, 245), np.random.randint(215, 245), np.random.randint(220, 250))
            cx, cy = np.random.randint(30, 220), np.random.randint(30, 220)
            r = np.random.randint(25, 70)
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=cloud_color)
            img = img.filter(ImageFilter.GaussianBlur(radius=8))

        img.save(f"{output_dir}/no_ships/no_ship_{i:04d}.png")

    # 2. Generate Ship Images
    for i in range(num_ship_samples):
        base_color = (
            np.random.randint(15, 45),
            np.random.randint(45, 95),
            np.random.randint(70, 140)
        )
        img = Image.new("RGB", image_size, color=base_color)
        draw = ImageDraw.Draw(img)

        # Waves
        for _ in range(np.random.randint(8, 20)):
            x1 = np.random.randint(0, image_size[0])
            y1 = np.random.randint(0, image_size[1])
            x2 = x1 + np.random.randint(15, 50)
            y2 = y1 + np.random.randint(-4, 4)
            wave_col = (
                min(255, base_color[0] + 20),
                min(255, base_color[1] + 25),
                min(255, base_color[2] + 35)
            )
            draw.line([(x1, y1), (x2, y2)], fill=wave_col, width=1)

        # Ship hull parameters
        cx = np.random.randint(60, image_size[0] - 60)
        cy = np.random.randint(60, image_size[1] - 60)
        length = np.random.randint(25, 65)
        width = np.random.randint(6, 16)
        angle_rad = np.random.uniform(0, 2 * np.pi)

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Ship Wake (white foamy trail behind ship)
        wake_len = length * 1.5
        wake_end_x = cx - wake_len * cos_a
        wake_end_y = cy - wake_len * sin_a
        wake_color = (200, 225, 240)
        draw.line([(cx, cy), (int(wake_end_x), int(wake_end_y))], fill=wake_color, width=width // 2)

        # Ship Hull Polygon (pointed front, rectangular stern)
        dx_l = (length / 2) * cos_a
        dy_l = (length / 2) * sin_a
        dx_w = (width / 2) * (-sin_a)
        dy_w = (width / 2) * cos_a

        bow = (int(cx + dx_l * 1.2), int(cy + dy_l * 1.2))
        port_bow = (int(cx + dx_l * 0.7 + dx_w), int(cy + dy_l * 0.7 + dy_w))
        port_stern = (int(cx - dx_l + dx_w), int(cy - dy_l + dy_w))
        stbd_stern = (int(cx - dx_l - dx_w), int(cy - dy_l - dy_w))
        stbd_bow = (int(cx + dx_l * 0.7 - dx_w), int(cy + dy_l * 0.7 - dy_w))

        hull_color = (
            np.random.randint(180, 240),  # White / metal grey / red ship hull
            np.random.randint(180, 230),
            np.random.randint(180, 230)
        )
        draw.polygon([bow, port_bow, port_stern, stbd_stern, stbd_bow], fill=hull_color)

        # Deck superstructure / container details
        deck_color = (np.random.randint(80, 160), np.random.randint(50, 100), np.random.randint(50, 100))
        draw.ellipse([(cx - width // 3, cy - width // 3), (cx + width // 3, cy + width // 3)], fill=deck_color)

        img.save(f"{output_dir}/ships/ship_{i:04d}.png")

    print(f"Sample dataset successfully generated in '{output_dir}'.")
    return {"ship_count": num_ship_samples, "no_ship_count": num_no_ship_samples, "path": output_dir}


def load_dataset_subset(
    data_dir: str = "data/sample_dataset",
    max_samples_per_class: Optional[int] = 500,
    target_size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads images and labels from data_dir.
    Searches for:
      - 'ships' and 'no_ships' subfolders (or 'ship' / 'no_ship')
    Limits loading to max_samples_per_class to support lightweight subset execution.
    Returns:
      images: np.ndarray of shape (N, H, W, 3) in RGB
      labels: np.ndarray of shape (N,) with 1 for ship, 0 for no_ship
      filepaths: list of loaded file paths
    """
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' not found. Auto-generating sample dataset...")
        generate_synthetic_samples(data_dir)

    ship_paths = sorted(glob.glob(f"{data_dir}/ships/*.png") + glob.glob(f"{data_dir}/ships/*.jpg") + glob.glob(f"{data_dir}/ship/*.jpg") + glob.glob(f"{data_dir}/ship/*.png"))
    no_ship_paths = sorted(glob.glob(f"{data_dir}/no_ships/*.png") + glob.glob(f"{data_dir}/no_ships/*.jpg") + glob.glob(f"{data_dir}/no_ship/*.jpg") + glob.glob(f"{data_dir}/no_ship/*.png"))

    if len(ship_paths) == 0 and len(no_ship_paths) == 0:
        print(f"No images found in '{data_dir}'. Generating synthetic subset...")
        generate_synthetic_samples(data_dir)
        ship_paths = sorted(glob.glob(f"{data_dir}/ships/*.png"))
        no_ship_paths = sorted(glob.glob(f"{data_dir}/no_ships/*.png"))

    if max_samples_per_class is not None:
        ship_paths = ship_paths[:max_samples_per_class]
        no_ship_paths = no_ship_paths[:max_samples_per_class]

    print(f"Loading {len(ship_paths)} ship images and {len(no_ship_paths)} no-ship images...")

    images = []
    labels = []
    filepaths = []

    for path in ship_paths:
        try:
            img = Image.open(path).convert('RGB')
            if img.size != target_size:
                img = img.resize(target_size)
            images.append(np.array(img, dtype=np.uint8))
            labels.append(1)
            filepaths.append(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    for path in no_ship_paths:
        try:
            img = Image.open(path).convert('RGB')
            if img.size != target_size:
                img = img.resize(target_size)
            images.append(np.array(img, dtype=np.uint8))
            labels.append(0)
            filepaths.append(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)

    # Shuffle dataset
    indices = np.arange(len(labels))
    np.random.seed(42)
    np.random.shuffle(indices)

    return images[indices], labels[indices], [filepaths[i] for i in indices]


def build_and_cache_features(
    images: np.ndarray,
    labels: np.ndarray,
    feature_set: str = 'all',
    cache_path: Optional[str] = "data/features_cache.npz"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts specified features for all images and optionally caches them to .npz
    """
    print(f"Extracting '{feature_set}' features for {len(images)} images...")
    X_features = []

    for i in range(len(images)):
        # Convert RGB to BGR for OpenCV compatibility if needed
        img_rgb = images[i]
        img_bgr = img_rgb[:, :, ::-1] if cv2 is not None else img_rgb
        feat = extract_features(img_bgr, feature_set=feature_set)
        X_features.append(feat)

    X_features = np.array(X_features, dtype=np.float32)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, X=X_features, y=labels, feature_set=feature_set)
        print(f"Cached features saved to '{cache_path}' (Shape: {X_features.shape})")

    return X_features, labels


if __name__ == '__main__':
    # Test generation and loading
    metadata = generate_synthetic_samples("data/sample_dataset", num_ship_samples=50, num_no_ship_samples=75)
    X_imgs, y_lbls, paths = load_dataset_subset("data/sample_dataset", max_samples_per_class=50)
    print(f"Loaded dataset: X shape = {X_imgs.shape}, y shape = {y_lbls.shape}")
    print(f"Class distribution: Ships = {np.sum(y_lbls == 1)}, No-Ships = {np.sum(y_lbls == 0)}")
