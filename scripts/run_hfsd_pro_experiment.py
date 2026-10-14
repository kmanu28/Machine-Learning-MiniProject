"""Tune and evaluate HFSD-Pro, a stronger CPU-only Airbus ship detector.

This is intentionally separate from HFSD so the original benchmark remains
reproducible. Model configurations, ensemble weights, and the threshold are
selected using validation data only; the test set is used exactly once.
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
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
    from skimage.feature import hog, local_binary_pattern
except ImportError:
    hog = None
    local_binary_pattern = None

RANDOM_STATE = 42


def resized_gray(image: np.ndarray, size: int = 128) -> np.ndarray:
    if cv2 is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    from PIL import Image
    gray = np.mean(image, axis=2).astype(np.uint8)
    return np.asarray(Image.fromarray(gray).resize((size, size)))


def fine_hog(image: np.ndarray) -> np.ndarray:
    gray = resized_gray(image)
    if hog is None:
        gx = np.gradient(gray.astype(np.float32), axis=1)
        gy = np.gradient(gray.astype(np.float32), axis=0)
        return np.asarray([np.mean(np.abs(gx)), np.mean(np.abs(gy)), np.std(gray)], dtype=np.float32)
    values = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                 cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    # Keep the fine descriptor compact enough for CPU classifiers.
    usable = values[:len(values) - (len(values) % 4)]
    return usable.reshape(-1, 4).mean(axis=1).astype(np.float32)


def local_texture_features(image: np.ndarray) -> np.ndarray:
    gray = resized_gray(image).astype(np.float32) / 255.0
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    edge = np.sqrt(gx * gx + gy * gy)
    values = []
    for row in range(4):
        for col in range(4):
            cell = gray[row * 32:(row + 1) * 32, col * 32:(col + 1) * 32]
            edge_cell = edge[row * 32:(row + 1) * 32, col * 32:(col + 1) * 32]
            values.extend([cell.mean(), cell.std(), np.percentile(cell, 10),
                           np.percentile(cell, 90), edge_cell.mean(), edge_cell.std(),
                           (edge_cell > 0.12).mean()])
            if local_binary_pattern is not None:
                lbp = local_binary_pattern(cell, P=8, R=1, method="uniform")
                hist, _ = np.histogram(lbp, bins=np.arange(11), range=(0, 10), density=True)
                values.extend(hist.astype(np.float32))
            else:
                values.extend(np.histogram(cell, bins=10, range=(0, 1), density=True)[0])
    return np.asarray(values, dtype=np.float32)


def color_quantiles(image: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] /= 180.0
        hsv[..., 1:] /= 255.0
    else:
        hsv = image.astype(np.float32) / 255.0
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    return np.asarray([np.quantile(hsv[..., channel], quantiles) for channel in range(3)], dtype=np.float32).ravel()


def extract_pro_features(images: np.ndarray, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)["X"]
    rows = []
    for index, image in enumerate(images, start=1):
        row = np.hstack((
            extract_features(image, "all"),
            fine_hog(image),
            local_texture_features(image),
            color_quantiles(image),
        ))
        rows.append(row)
        if index % 500 == 0 or index == len(images):
            print(f"  extracted {index}/{len(images)} images", flush=True)
    features = np.asarray(rows, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=features)
    return features


def score(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    pred = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def best_threshold(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict]:
    best = None
    for threshold in np.arange(0.10, 0.91, 0.01):
        current = score(y, probabilities, float(threshold))
        if best is None or (current["f1"], current["accuracy"]) > (best[1]["f1"], best[1]["accuracy"]):
            best = (float(threshold), current)
    return best


def fit_and_predict(model, X_train, y_train, weights, X_val, X_test):
    model.fit(X_train, y_train, sample_weight=weights)
    return model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune/evaluate CPU-only HFSD-Pro.")
    parser.add_argument("--data-dir", default="data/subset_6000")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--cache-dir", default="data/hfsd_pro_cache")
    args = parser.parse_args()

    data_root, output_root, cache_root = Path(args.data_dir), Path(args.output_dir), Path(args.cache_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    split_data, X = {}, {}
    for split in ("train", "val", "test"):
        images, labels, _ = load_dataset_subset(str(data_root / split), max_samples_per_class=None)
        split_data[split] = (images, labels)
        print(f"{split}: {len(labels)} images ({int(labels.sum())} ships)", flush=True)
        print(f"Extracting HFSD-Pro features for {split}...", flush=True)
        X[split] = extract_pro_features(images, cache_root / f"{split}.npz")
    print(f"Feature dimension: {X['train'].shape[1]}", flush=True)

    y_train, y_val, y_test = (split_data[s][1] for s in ("train", "val", "test"))
    class_weights = np.where(y_train == 1,
                             len(y_train) / (2.0 * max(1, int(y_train.sum()))),
                             len(y_train) / (2.0 * max(1, int((y_train == 0).sum()))))
    scaler = StandardScaler().fit(X["train"])
    candidate_probabilities = {}
    selected_models = {}

    hgb_configs = [
        ("hgb_a", 0.04, 15, 10, 350),
        ("hgb_b", 0.05, 31, 12, 300),
        ("hgb_c", 0.03, 63, 15, 400),
    ]
    best_hgb = None
    for name, learning_rate, leaves, min_leaf, iterations in hgb_configs:
        model = HistGradientBoostingClassifier(learning_rate=learning_rate, max_leaf_nodes=leaves,
                                                min_samples_leaf=min_leaf, max_iter=iterations,
                                                l2_regularization=1.0, random_state=RANDOM_STATE)
        val_prob, test_prob = fit_and_predict(model, X["train"], y_train, class_weights, X["val"], X["test"])
        threshold, validation_score = best_threshold(y_val, val_prob)
        print(f"{name}: val F1={validation_score['f1']:.4f} threshold={threshold:.2f}", flush=True)
        if best_hgb is None or (validation_score["f1"], validation_score["accuracy"]) > (best_hgb[0]["f1"], best_hgb[0]["accuracy"]):
            best_hgb = (validation_score, model, val_prob, test_prob)
    selected_models["hgb"] = best_hgb[1]
    candidate_probabilities["hgb"] = (best_hgb[2], best_hgb[3])

    rf_configs = [
        ("rf_a", 300, 16, 1),
        ("rf_b", 400, 22, 2),
    ]
    best_rf = None
    for name, trees, depth, leaf in rf_configs:
        model = RandomForestClassifier(n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
                                       class_weight="balanced_subsample", n_jobs=-1,
                                       random_state=RANDOM_STATE)
        val_prob, test_prob = fit_and_predict(model, X["train"], y_train, class_weights, X["val"], X["test"])
        threshold, validation_score = best_threshold(y_val, val_prob)
        print(f"{name}: val F1={validation_score['f1']:.4f} threshold={threshold:.2f}", flush=True)
        if best_rf is None or (validation_score["f1"], validation_score["accuracy"]) > (best_rf[0]["f1"], best_rf[0]["accuracy"]):
            best_rf = (validation_score, model, val_prob, test_prob)
    selected_models["rf"] = best_rf[1]
    candidate_probabilities["rf"] = (best_rf[2], best_rf[3])

    best_logistic = None
    for C in (0.15, 0.5, 1.2):
        model = LogisticRegression(C=C, class_weight="balanced", max_iter=1500,
                                   solver="liblinear", random_state=RANDOM_STATE)
        model.fit(scaler.transform(X["train"]), y_train)
        val_prob = model.predict_proba(scaler.transform(X["val"]))[:, 1]
        test_prob = model.predict_proba(scaler.transform(X["test"]))[:, 1]
        threshold, validation_score = best_threshold(y_val, val_prob)
        print(f"logistic_C_{C}: val F1={validation_score['f1']:.4f} threshold={threshold:.2f}", flush=True)
        if best_logistic is None or (validation_score["f1"], validation_score["accuracy"]) > (best_logistic[0]["f1"], best_logistic[0]["accuracy"]):
            best_logistic = (validation_score, model, val_prob, test_prob)
    selected_models["logistic"] = best_logistic[1]
    candidate_probabilities["logistic"] = (best_logistic[2], best_logistic[3])

    # Tune blend weights on validation only. The grid is deliberately small to
    # keep the method reproducible and avoid a large hyperparameter search.
    best_blend = None
    for hgb_weight in np.arange(0.0, 1.01, 0.1):
        for rf_weight in np.arange(0.0, 1.01 - hgb_weight, 0.1):
            logistic_weight = round(1.0 - hgb_weight - rf_weight, 2)
            val_prob = (hgb_weight * candidate_probabilities["hgb"][0]
                        + rf_weight * candidate_probabilities["rf"][0]
                        + logistic_weight * candidate_probabilities["logistic"][0])
            threshold, validation_score = best_threshold(y_val, val_prob)
            key = (validation_score["f1"], validation_score["accuracy"])
            if best_blend is None or key > (best_blend[0]["f1"], best_blend[0]["accuracy"]):
                best_blend = (validation_score, {"hgb": round(float(hgb_weight), 2), "rf": round(float(rf_weight), 2), "logistic": logistic_weight}, threshold)

    weights = best_blend[1]
    threshold = best_blend[2]
    val_prob = sum(weights[name] * candidate_probabilities[name][0] for name in weights)
    test_prob = sum(weights[name] * candidate_probabilities[name][1] for name in weights)
    train_prob = None
    result = {
        "method": "HFSD-Pro - Tuned Hybrid Feature-Fusion Ship Detector",
        "description": "Existing 531 features + compact fine HOG + local texture/LBP statistics + HSV quantiles; validation-tuned HGB/RF/logistic ensemble.",
        "dataset": {"train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test)),
                    "train_ships": int(y_train.sum()), "val_ships": int(y_val.sum()), "test_ships": int(y_test.sum())},
        "feature_dimension": int(X["train"].shape[1]),
        "selected_blend_weights": weights,
        "validation_threshold": float(threshold),
        "validation_metrics": score(y_val, val_prob, threshold),
        "test_metrics": score(y_test, test_prob, threshold),
        "test_metrics_at_0_5": score(y_test, test_prob, 0.5),
    }
    with (output_root / "hfsd_pro_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    joblib.dump({"models": selected_models, "scaler": scaler, "threshold": threshold,
                 "blend_weights": weights, "feature_dimension": int(X["train"].shape[1])},
                output_root / "hfsd_pro_pipeline.joblib")
    print(json.dumps(result["test_metrics"], indent=2))
    print(f"Saved {output_root / 'hfsd_pro_results.json'}")
    print(f"Saved {output_root / 'hfsd_pro_pipeline.joblib'}")


if __name__ == "__main__":
    main()
