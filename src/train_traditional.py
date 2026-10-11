"""
Training and Comprehensive Evaluation Script for Traditional ML Models
Reproduces Table 1 & Figure 5/6 from Stanford CS229 Airbus Ship Detection paper:
- Feature set comparison: His, Ha, Hu, His+Ha, Ha+Hu, Hu+His, w/ All, w/o (Raw)
- Algorithms: LDA, KNN, Naive Bayes, Random Forest, SVM
- 10-Fold Stratified Cross Validation
- Model Serialization & Metric Export
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_dataset_subset, build_and_cache_features
from src.features import extract_features
from src.models_traditional import (
    get_traditional_classifiers,
    evaluate_model_metrics,
    train_and_evaluate_cv,
    scan_thresholds,
    save_trained_pipeline
)


def run_feature_combination_experiments(
    images: np.ndarray,
    labels: np.ndarray,
    feature_sets = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates all ML algorithms across all feature engineering configurations.
    Returns comparison DataFrame mirroring Table 1 in the CS229 paper.
    """
    if feature_sets is None:
        feature_sets = ['his', 'ha', 'hu', 'his+ha', 'ha+hu', 'hu+his', 'all', 'raw']

    print(f"\n=======================================================")
    print(f"Starting Feature Engineering & ML Comparison Experiment")
    print(f"Total samples: {len(images)} (Ships: {np.sum(labels==1)}, No-Ships: {np.sum(labels==0)})")
    print(f"=======================================================\n")

    # Split into Train and Test sets
    img_train, img_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    results_table = {}

    for feat_name in feature_sets:
        print(f"--> Extracting & evaluating feature set: '{feat_name}'...")
        X_train, _ = build_and_cache_features(img_train, y_train, feature_set=feat_name, cache_path=None)
        X_test, _ = build_and_cache_features(img_test, y_test, feature_set=feat_name, cache_path=None)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = get_traditional_classifiers()
        col_name = "w/ All" if feat_name == "all" else ("w/o" if feat_name == "raw" else feat_name.capitalize())
        results_table[col_name] = {}

        for alg_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            metrics = evaluate_model_metrics(y_test, y_pred)
            acc = round(metrics["accuracy"] * 100, 2)
            results_table[col_name][alg_name] = f"{acc}%"

            # Save the 'all' feature models for live demo deployment
            if feat_name == 'all':
                save_path = save_trained_pipeline(model, scaler, feat_name, save_dir="models")

    df_results = pd.DataFrame(results_table)
    return df_results


def run_full_training_pipeline(
    data_dir: str = "data/sample_dataset",
    max_samples_per_class: int = 250,
    output_report_path: str = "models/experiment_results.json"
):
    """
    Executes full pipeline:
    1. Loads dataset subset
    2. Runs feature combination benchmarks
    3. Runs 10-fold cross validation for stability analysis
    4. Performs decision threshold scanning on Random Forest
    5. Saves metrics summary
    """
    os.makedirs("models", exist_ok=True)

    # 1. Load data
    images, labels, paths = load_dataset_subset(data_dir=data_dir, max_samples_per_class=max_samples_per_class)

    # 2. Benchmark feature engineering matrix
    df_comparison = run_feature_combination_experiments(images, labels)
    
    print("\n=======================================================")
    print("Table 1: Test Accuracy Comparison across Feature Sets:")
    print("=======================================================")
    print(df_comparison.to_string())
    print("=======================================================\n")

    # 3. 10-Fold Cross-Validation on 'all' features
    print("--> Running 10-Fold Stratified Cross Validation on 'All Features'...")
    X_all, _ = build_and_cache_features(images, labels, feature_set='all', cache_path="data/features_all.npz")
    cv_results = train_and_evaluate_cv(X_all, labels, n_splits=10)

    print("\n10-Fold CV Summary (Accuracy % +- Std %):")
    for alg, res in cv_results.items():
        mean_pct = round(res['accuracy_mean'] * 100, 2)
        std_pct = round(res['accuracy_std'] * 100, 2)
        print(f"  {alg:5s}: {mean_pct:6.2f}% (+- {std_pct:.2f}%) | Prec: {res['precision_mean']*100:.1f}% | Rec: {res['recall_mean']*100:.1f}%")

    # 4. Threshold Optimization for Random Forest
    scaler = StandardScaler()
    X_train, X_val, y_train, y_val = train_test_split(X_all, labels, test_size=0.2, stratify=labels, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    
    rf = get_traditional_classifiers()["RF"]
    rf.fit(X_train_scaled, y_train)
    thresh_results = scan_thresholds(rf, scaler, X_val, y_val)
    print(f"\nRandom Forest Optimal Decision Threshold: {thresh_results['best_threshold']} (Best F1: {thresh_results['best_f1']:.4f})")

    # Save summary report to JSON
    summary = {
        "dataset_size": len(images),
        "ship_samples": int(np.sum(labels == 1)),
        "no_ship_samples": int(np.sum(labels == 0)),
        "table_1_comparison": df_comparison.to_dict(),
        "cross_validation_10fold": cv_results,
        "rf_threshold_optimization": thresh_results
    }

    with open(output_report_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n[OK] Training complete. Models saved to 'models/' and metrics saved to '{output_report_path}'.")
    return summary


if __name__ == '__main__':
    run_full_training_pipeline()
