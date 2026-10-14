"""Run the CPU-friendly Hybrid Feature-Fusion Ship Detector (HFSD).

HFSD combines the project's existing HSV histogram, Haralick texture, and Hu
shape features with a compact HOG descriptor and spatial HSV statistics.  It
then blends three CPU-friendly classifiers and tunes the decision threshold on
the validation split only.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_dataset_subset
from src.features import extract_features

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from skimage.feature import hog
except ImportError:
    hog = None


RANDOM_STATE = 42
BLEND_WEIGHTS = {"hgb": 0.55, "rf": 0.30, "logistic": 0.15}


def compact_hog(image: np.ndarray) -> np.ndarray:
    """Return a small rotation-tolerant-ish edge descriptor for CPU training."""
    if cv2 is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    else:
        gray = np.mean(image, axis=2).astype(np.uint8)
        from PIL import Image
        gray = np.asarray(Image.fromarray(gray).resize((128, 128)))

    if hog is not None:
        values = hog(
            gray,
            orientations=9,
            pixels_per_cell=(32, 32),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return values.astype(np.float32)

    # Fallback: gradient magnitude/orientation histograms over a 4x4 grid.
    gx = np.gradient(gray.astype(np.float32), axis=1)
    gy = np.gradient(gray.astype(np.float32), axis=0)
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = (np.arctan2(gy, gx) + np.pi) * (9.0 / (2.0 * np.pi))
    result = []
    for row in range(4):
        for col in range(4):
            ys = slice(row * 32, (row + 1) * 32)
            xs = slice(col * 32, (col + 1) * 32)
            hist = np.zeros(9, dtype=np.float32)
            bins = np.clip(angle[ys, xs].astype(np.int32), 0, 8)
            np.add.at(hist, bins.ravel(), magnitude[ys, xs].ravel())
            result.extend(hist / (np.linalg.norm(hist) + 1e-8))
    return np.asarray(result, dtype=np.float32)


def spatial_hsv_statistics(image: np.ndarray) -> np.ndarray:
    """Capture where colors/textures occur, not just their global frequency."""
    if cv2 is not None:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] /= 180.0
        hsv[..., 1:] /= 255.0
    else:
        rgb = image.astype(np.float32) / 255.0
        mx, mn = rgb.max(axis=2), rgb.min(axis=2)
        delta = mx - mn + 1e-8
        sat = np.where(mx > 0, delta / (mx + 1e-8), 0)
        hue = np.zeros_like(mx)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mask = delta > 1e-7
        hue[mask & (mx == r)] = ((g[mask & (mx == r)] - b[mask & (mx == r)]) / delta[mask & (mx == r)]) % 6
        hue[mask & (mx == g)] = ((b[mask & (mx == g)] - r[mask & (mx == g)]) / delta[mask & (mx == g)]) + 2
        hue[mask & (mx == b)] = ((r[mask & (mx == b)] - g[mask & (mx == b)]) / delta[mask & (mx == b)]) + 4
        hsv = np.stack([(hue / 6.0) % 1.0, sat, mx], axis=2)

    h, w = hsv.shape[:2]
    values = []
    for row in range(4):
        for col in range(4):
            cell = hsv[row * h // 4:(row + 1) * h // 4,
                       col * w // 4:(col + 1) * w // 4]
            values.extend(cell.mean(axis=(0, 1)))
            values.extend(cell.std(axis=(0, 1)))
    return np.asarray(values, dtype=np.float32)


def extract_hfsd_features(images: np.ndarray, cache_path: Path | None = None) -> np.ndarray:
    """Extract and optionally cache deterministic features for a split."""
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)["X"]

    rows = []
    for index, image in enumerate(images, start=1):
        base = extract_features(image, "all")
        row = np.hstack((base, compact_hog(image), spatial_hsv_statistics(image)))
        rows.append(row)
        if index % 500 == 0 or index == len(images):
            print(f"  extracted {index}/{len(images)} images")
    features = np.asarray(rows, dtype=np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=features)
    return features


def metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, list[dict]]:
    scan = []
    best = (0.5, -1.0, -1.0)
    for threshold in np.arange(0.10, 0.91, 0.02):
        current = metrics(y_true, probabilities, float(threshold))
        scan.append(current)
        # Maximize F1, then accuracy as a deterministic tie-breaker.
        score = (current["f1"], current["accuracy"])
        if score > (best[1], best[2]):
            best = (float(threshold), current["f1"], current["accuracy"])
    return best[0], scan


def blend(models: dict, scaler: StandardScaler, features: np.ndarray) -> np.ndarray:
    scaled = scaler.transform(features)
    probabilities = (
        BLEND_WEIGHTS["hgb"] * models["hgb"].predict_proba(features)[:, 1]
        + BLEND_WEIGHTS["rf"] * models["rf"].predict_proba(features)[:, 1]
        + BLEND_WEIGHTS["logistic"] * models["logistic"].predict_proba(scaled)[:, 1]
    )
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate CPU-friendly HFSD.")
    parser.add_argument("--data-dir", default="data/subset_6000")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--cache-dir", default="data/hfsd_cache")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split in ("train", "val", "test"):
        images, labels, _ = load_dataset_subset(str(data_dir / split), max_samples_per_class=None)
        splits[split] = (images, labels)
        print(f"{split}: {len(labels)} images ({int(labels.sum())} ships)")

    features = {}
    for split, (images, _) in splits.items():
        print(f"Extracting HFSD features for {split}...")
        features[split] = extract_hfsd_features(images, cache_dir / f"{split}.npz")
    print(f"Feature dimension: {features['train'].shape[1]}")

    y_train, y_val, y_test = (splits[name][1] for name in ("train", "val", "test"))
    X_train, X_val, X_test = (features[name] for name in ("train", "val", "test"))

    positive_weight = len(y_train) / (2.0 * max(1, int(y_train.sum())))
    negative_weight = len(y_train) / (2.0 * max(1, int((y_train == 0).sum())))
    sample_weights = np.where(y_train == 1, positive_weight, negative_weight)

    models = {
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
            min_samples_leaf=12, l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=18, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "logistic": LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=1200,
            solver="liblinear", random_state=RANDOM_STATE,
        ),
    }
    scaler = StandardScaler().fit(X_train)
    for name in ("hgb", "rf"):
        print(f"Training {name}...")
        models[name].fit(X_train, y_train, sample_weight=sample_weights)
    print("Training logistic...")
    models["logistic"].fit(scaler.transform(X_train), y_train)

    val_probabilities = blend(models, scaler, X_val)
    threshold, threshold_scan = choose_threshold(y_val, val_probabilities)
    test_probabilities = blend(models, scaler, X_test)
    train_probabilities = blend(models, scaler, X_train)

    result = {
        "method": "HFSD - Hybrid Feature-Fusion Ship Detector",
        "description": "HSV histogram + Haralick + Hu moments + compact HOG + 4x4 spatial HSV statistics; HGB/RF/logistic probability blend.",
        "dataset": {
            "train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test)),
            "train_ships": int(y_train.sum()), "val_ships": int(y_val.sum()), "test_ships": int(y_test.sum()),
        },
        "feature_dimension": int(X_train.shape[1]),
        "blend_weights": BLEND_WEIGHTS,
        "validation_threshold": float(threshold),
        "train_metrics": metrics(y_train, train_probabilities, threshold),
        "validation_metrics": metrics(y_val, val_probabilities, threshold),
        "test_metrics": metrics(y_test, test_probabilities, threshold),
        "test_metrics_at_0_5": metrics(y_test, test_probabilities, 0.5),
        "threshold_scan": threshold_scan,
    }
    with (output_dir / "hfsd_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    joblib.dump({"models": models, "scaler": scaler, "threshold": threshold,
                 "feature_dimension": int(X_train.shape[1]), "blend_weights": BLEND_WEIGHTS},
                output_dir / "hfsd_pipeline.joblib")

    thresholds = [row["threshold"] for row in threshold_scan]
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, [row["f1"] for row in threshold_scan], label="F1")
    plt.plot(thresholds, [row["accuracy"] for row in threshold_scan], label="Accuracy")
    plt.axvline(threshold, color="black", linestyle="--", label=f"Selected: {threshold:.2f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title("HFSD validation threshold selection")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hfsd_threshold_scan.png", dpi=180)
    plt.close()

    print(json.dumps(result["test_metrics"], indent=2))
    print(f"Saved {output_dir / 'hfsd_results.json'}")
    print(f"Saved {output_dir / 'hfsd_pipeline.joblib'}")


if __name__ == "__main__":
    main()
