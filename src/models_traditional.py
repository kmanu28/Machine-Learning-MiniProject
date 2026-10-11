"""
Traditional Machine Learning Models for Airbus Ship Detection:
1. Linear Discriminant Analysis (LDA)
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes (NB)
4. Random Forest Classifier (RF - 70 estimators as per paper)
5. Support Vector Machine (SVM - RBF kernel)

Includes:
- 10-Fold Cross-Validation evaluation
- Threshold scanning for imbalanced data optimization
- Precision, Recall, Specificity, F1, Accuracy, and Confusion Matrix metrics
"""

import numpy as np
import os
import sys
import joblib
from typing import Dict, Any, Tuple, List

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_traditional_classifiers() -> Dict[str, Any]:
    """Returns dictionary of baseline ML classifiers with configurations aligned with the paper."""
    return {
        "LDA": LinearDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "NB": GaussianNB(),
        "RF": RandomForestClassifier(n_estimators=70, max_depth=15, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    }


def evaluate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """Calculates all key evaluation metrics: Accuracy, Precision, Recall, Specificity, F1, AUC."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity / True Positive Rate
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0     # Specificity / True Negative Rate
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if y_proba is not None and len(np.unique(y_true)) > 1:
        proba_1d = y_proba[:, 1] if (len(y_proba.shape) > 1 and y_proba.shape[1] > 1) else y_proba
        try:
            auc = roc_auc_score(y_true, proba_1d)
        except Exception:
            auc = 0.0
    else:
        auc = 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }


def train_and_evaluate_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Performs 10-Fold Stratified Cross Validation across all traditional ML classifiers.
    Returns mean and std of accuracy and metrics for each algorithm.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = get_traditional_classifiers()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {}

    for name, model in models.items():
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_specificities = []
        fold_f1s = []

        for train_idx, val_idx in skf.split(X_scaled, y):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = f1_score(y_val, y_pred, zero_division=0)

            fold_accuracies.append(acc)
            fold_precisions.append(prec)
            fold_recalls.append(rec)
            fold_specificities.append(spec)
            fold_f1s.append(f1)

        results[name] = {
            "accuracy_mean": float(np.mean(fold_accuracies)),
            "accuracy_std": float(np.std(fold_accuracies)),
            "precision_mean": float(np.mean(fold_precisions)),
            "recall_mean": float(np.mean(fold_recalls)),
            "specificity_mean": float(np.mean(fold_specificities)),
            "f1_mean": float(np.mean(fold_f1s)),
            "fold_accuracies": [float(a) for a in fold_accuracies]
        }

    return results


def scan_thresholds(
    model: Any,
    scaler: StandardScaler,
    X_val: np.ndarray,
    y_val: np.ndarray,
    thresholds: List[float] = None
) -> Dict[str, Any]:
    """
    Scans decision thresholds from 0.1 to 0.9 to optimize accuracy & F1 on imbalanced data.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 0.95, 0.05)]

    X_val_scaled = scaler.transform(X_val)
    probas = model.predict_proba(X_val_scaled)[:, 1]

    best_thresh = 0.5
    best_f1 = -1.0
    scan_results = []

    for t in thresholds:
        y_pred = (probas >= t).astype(int)
        metrics = evaluate_model_metrics(y_val, y_pred, probas)
        scan_results.append({
            "threshold": t,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thresh = t

    return {
        "best_threshold": best_thresh,
        "best_f1": best_f1,
        "threshold_curve": scan_results
    }


def save_trained_pipeline(model: Any, scaler: StandardScaler, feature_set: str, save_dir: str = "models") -> str:
    """Saves trained model and scaler to disk."""
    os.makedirs(save_dir, exist_ok=True)
    model_name = model.__class__.__name__
    filepath = os.path.join(save_dir, f"{model_name}_{feature_set}.joblib")
    joblib.dump({"model": model, "scaler": scaler, "feature_set": feature_set}, filepath)
    return filepath


def load_trained_pipeline(filepath: str) -> Dict[str, Any]:
    """Loads saved model, scaler and feature settings."""
    return joblib.load(filepath)
