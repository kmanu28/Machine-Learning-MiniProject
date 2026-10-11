"""
Visualization and Diagnostic Charts for Airbus Ship Detection:
1. Feature visualizer: plots original image, Grayscale, HSV channels, GLCM properties, Hu Moments
2. Model Comparison Bar Chart (Table 1 comparison)
3. 10-Fold Cross Validation Box Plot (Figure 5 from paper)
4. Decision Threshold Scanning Curve (Figure 7 from paper)
5. Normalized Confusion Matrices (Figure 8 from paper)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract_hu_moments, extract_color_histogram, extract_haralick_textures, rgb_to_gray, rgb_to_hsv_manual


def plot_feature_breakdown(image_rgb: np.ndarray, title: str = "Feature Breakdown", save_path: str = "models/feature_visualization.png"):
    """
    Plots comprehensive multi-panel breakdown of all extracted features for an image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    hu = extract_hu_moments(image_rgb)
    hsv = rgb_to_hsv_manual(image_rgb)
    gray = rgb_to_gray(image_rgb)
    ha = extract_haralick_textures(image_rgb)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Panel 1: Original Image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Input Aerial Image (256x256)")
    axes[0, 0].axis('off')

    # Panel 2: Grayscale
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title("Grayscale (Hu & Texture base)")
    axes[0, 1].axis('off')

    # Panel 3: HSV Hue Channel
    axes[0, 2].imshow(hsv[:, :, 0], cmap='hsv')
    axes[0, 2].set_title("HSV Color Space (Hue Channel)")
    axes[0, 2].axis('off')

    # Panel 4: Hu Moments (7 invariants)
    axes[1, 0].bar([f"H{i+1}" for i in range(len(hu))], hu, color='#2b5c8f', edgecolor='black')
    axes[1, 0].set_title("Hu Invariant Shape Moments (-log10 scale)")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # Panel 5: Haralick GLCM Texture Features
    labels = ['Contrast', 'Dissim', 'Homo', 'Energy', 'Corr', 'ASM']
    vals = [ha[0], ha[2], ha[4], ha[6], ha[8], ha[10]]
    axes[1, 1].bar(labels, vals, color='#3a9278', edgecolor='black')
    axes[1, 1].set_title("Haralick GLCM Textures (Mean)")
    axes[1, 1].tick_params(axis='x', rotation=30)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # Panel 6: HSV Color Histogram Profile
    hist = extract_color_histogram(image_rgb)
    axes[1, 2].plot(hist[:64], color='#d95f02', lw=2)
    axes[1, 2].set_title("HSV 3D Color Histogram (First 64 bins)")
    axes[1, 2].set_xlabel("Bin Index")
    axes[1, 2].set_ylabel("Normalized Frequency")
    axes[1, 2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Feature breakdown saved to '{save_path}'")


def plot_cross_validation_boxplot(cv_results: dict, save_path: str = "models/cv_boxplot.png"):
    """
    Plots Figure 5 from paper: Box Plot for 10-Fold Test Accuracy of ML Algorithms.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    algs = list(cv_results.keys())
    data = [cv_results[alg]["fold_accuracies"] for alg in algs]
    # Multiply by 100 for percentage
    data_pct = [[acc * 100 for acc in fold] for fold in data]

    plt.figure(figsize=(9, 6))
    sns.set_style("whitegrid")
    
    palette = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974']
    try:
        box = plt.boxplot(data_pct, tick_labels=algs, patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
    except TypeError:
        box = plt.boxplot(data_pct, labels=algs, patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
    
    for patch, color in zip(box['boxes'], palette[:len(algs)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    plt.title("10-Fold Cross Validation Accuracy Comparison (CS229 Fig 5)", fontsize=14, fontweight='bold', pad=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.xlabel("Machine Learning Algorithm", fontsize=12)
    plt.ylim(50, 102)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"CV Box Plot saved to '{save_path}'")


def plot_threshold_curve(threshold_data: list, save_path: str = "models/threshold_scanning.png"):
    """
    Plots Figure 7 from paper: Threshold Scanning (Accuracy, Precision, Recall, F1 vs Threshold).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    threshs = [x["threshold"] for x in threshold_data]
    accs = [x["accuracy"] * 100 for x in threshold_data]
    precs = [x["precision"] * 100 for x in threshold_data]
    recs = [x["recall"] * 100 for x in threshold_data]
    f1s = [x["f1"] * 100 for x in threshold_data]

    plt.figure(figsize=(9, 6))
    plt.plot(threshs, accs, label="Accuracy", color="#1f77b4", lw=2.5)
    plt.plot(threshs, precs, label="Precision", color="#2ca02c", lw=2, linestyle="--")
    plt.plot(threshs, recs, label="Recall (Sensitivity)", color="#d62728", lw=2, linestyle="-.")
    plt.plot(threshs, f1s, label="F1-Score", color="#9467bd", lw=2.5)

    plt.title("Decision Threshold Scanning for Imbalanced Data (CS229 Fig 7)", fontsize=14, fontweight='bold')
    plt.xlabel("Classification Decision Threshold", fontsize=12)
    plt.ylabel("Score (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", frameon=True)
    plt.ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Threshold Scanning curve saved to '{save_path}'")


def plot_confusion_matrix_heatmap(cm: list, model_name: str = "Random Forest", save_path: str = "models/confusion_matrix.png"):
    """
    Plots normalized confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_arr / row_sums

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=["Predicted No Ship", "Predicted Ship"],
        yticklabels=["Actual No Ship", "Actual Ship"],
        cbar=True, annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title(f"Normalized Confusion Matrix - {model_name}", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Confusion Matrix saved to '{save_path}'")


def generate_all_visualizations(report_path: str = "models/experiment_results.json"):
    """Loads experiment results and generates all figures."""
    if not os.path.exists(report_path):
        print(f"Report file '{report_path}' not found.")
        return

    with open(report_path, "r") as f:
        data = json.load(f)

    if "cross_validation_10fold" in data:
        plot_cross_validation_boxplot(data["cross_validation_10fold"])

    if "rf_threshold_optimization" in data and "threshold_curve" in data["rf_threshold_optimization"]:
        plot_threshold_curve(data["rf_threshold_optimization"]["threshold_curve"])

    # Sample image feature breakdown
    sample_ship = "data/sample_dataset/ships/ship_0000.png"
    if os.path.exists(sample_ship):
        from PIL import Image
        img = np.array(Image.open(sample_ship).convert('RGB'))
        plot_feature_breakdown(img, title="Feature Extraction Analysis (Ship Target)", save_path="models/ship_feature_analysis.png")

    sample_no_ship = "data/sample_dataset/no_ships/no_ship_0000.png"
    if os.path.exists(sample_no_ship):
        from PIL import Image
        img = np.array(Image.open(sample_no_ship).convert('RGB'))
        plot_feature_breakdown(img, title="Feature Extraction Analysis (Sea / Non-Ship)", save_path="models/no_ship_feature_analysis.png")


if __name__ == '__main__':
    generate_all_visualizations()
