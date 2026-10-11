"""Run the full traditional-ML experiment on a prepared Airbus subset."""
import argparse
import json
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import load_dataset_subset, build_and_cache_features
from src.models_traditional import (
    get_traditional_classifiers, evaluate_model_metrics,
    train_and_evaluate_cv, scan_thresholds, save_trained_pipeline,
)

FEATURE_SETS = ["his", "ha", "hu", "his+ha", "ha+hu", "hu+his", "all", "raw"]


def load_split(root, split):
    return load_dataset_subset(os.path.join(root, split), max_samples_per_class=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/subset_6000")
    ap.add_argument("--output", default="models/experiment_results.json")
    args = ap.parse_args()

    train_images, y_train, _ = load_split(args.data_dir, "train")
    val_images, y_val, _ = load_split(args.data_dir, "val")
    test_images, y_test, _ = load_split(args.data_dir, "test")
    print(f"Train={len(y_train)}, validation={len(y_val)}, test={len(y_test)}")

    table = {}
    for feature_set in FEATURE_SETS:
        print(f"Extracting {feature_set} features...")
        X_train, _ = build_and_cache_features(train_images, y_train, feature_set, cache_path=None)
        X_test, _ = build_and_cache_features(test_images, y_test, feature_set, cache_path=None)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        table[feature_set] = {}
        for name, model in get_traditional_classifiers().items():
            model.fit(X_train_scaled, y_train)
            prediction = model.predict(X_test_scaled)
            probability = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None
            table[feature_set][name] = evaluate_model_metrics(y_test, prediction, probability)
            if feature_set == "all":
                save_trained_pipeline(model, scaler, feature_set, save_dir="models")

    X_all, _ = build_and_cache_features(train_images, y_train, "all", cache_path="data/features_train.npz")
    cv = train_and_evaluate_cv(X_all, y_train, n_splits=10)
    X_val, _ = build_and_cache_features(val_images, y_val, "all", cache_path=None)
    scaler = StandardScaler().fit(X_all)
    rf = get_traditional_classifiers()["RF"]
    rf.fit(scaler.transform(X_all), y_train)
    threshold = scan_thresholds(rf, scaler, X_val, y_val)

    summary = {
        "dataset": {"train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test)),
                    "train_ships": int(y_train.sum()), "val_ships": int(y_val.sum()), "test_ships": int(y_test.sum())},
        "table_1_comparison": table,
        "cross_validation_10fold_train": cv,
        "rf_threshold_optimization_validation": threshold,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()