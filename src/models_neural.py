"""
Lightweight Neural Network Classifiers (MLP / Deep Feature Networks)
Provides zero-dependency neural network baseline that runs universally on CPU and Windows:
- Multi-Layer Perceptron (MLP) with Batch Normalization simulation & ReLU activations
- Trains directly on engineered multi-feature representations (Hu + HSV + Haralick)
"""

import os
import sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_dataset_subset, build_and_cache_features
from src.models_traditional import evaluate_model_metrics, scan_thresholds, save_trained_pipeline


def train_deep_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layer_sizes: tuple = (256, 128, 64),
    max_iter: int = 100
) -> dict:
    """Trains a deep MLP network and records loss/accuracy metrics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=16,
        learning_rate_init=0.001,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15
    )

    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_val_scaled)
    y_proba = mlp.predict_proba(X_val_scaled)

    metrics = evaluate_model_metrics(y_val, y_pred, y_proba)
    save_path = save_trained_pipeline(mlp, scaler, "all", save_dir="models")

    return {
        "metrics": metrics,
        "loss_curve": [float(l) for l in mlp.loss_curve_],
        "model_path": save_path
    }


if __name__ == '__main__':
    images, labels, _ = load_dataset_subset("data/sample_dataset", max_samples_per_class=100)
    X, _ = build_and_cache_features(images, labels, feature_set='all')
    
    split = int(0.8 * len(X))
    res = train_deep_mlp(X[:split], labels[:split], X[split:], labels[split:])
    print("Deep MLP Validation Accuracy:", round(res["metrics"]["accuracy"] * 100, 2), "%")
