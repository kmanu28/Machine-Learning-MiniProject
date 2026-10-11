"""
Interactive Live Demonstration Web Application for Airbus Ship Detection
Provides:
1. Drag-and-drop Satellite Image Classification
2. Real-time Feature Extraction Visualizations (Hu Moments, HSV Histogram, Haralick Textures)
3. Side-by-Side Model Comparison (Random Forest, SVM, KNN, LDA, Naive Bayes, CNN)
4. Presets and Sample Gallery
"""

import os
import io
import sys
import json
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory

# Ensure project root in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.features import (
    extract_features,
    extract_hu_moments,
    extract_color_histogram,
    extract_haralick_textures,
    rgb_to_gray,
    rgb_to_hsv_manual
)
from src.models_traditional import load_trained_pipeline, get_traditional_classifiers

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

# Load available pre-trained models
MODELS = {}
MODEL_DIR = os.path.join(BASE_DIR, "models")


def get_available_models():
    """Loads all serialized models from models directory."""
    loaded = {}
    if os.path.exists(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith(".joblib") and "all" in f:
                path = os.path.join(MODEL_DIR, f)
                name = f.split("_")[0]
                try:
                    loaded[name] = load_trained_pipeline(path)
                except Exception as e:
                    print(f"Error loading {f}: {e}")
    return loaded


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns sample image URLs from the dataset for quick testing."""
    sample_dir = os.path.join(BASE_DIR, "data", "sample_dataset")
    samples = []

    for category in ["ships", "no_ships"]:
        cat_dir = os.path.join(sample_dir, category)
        if os.path.exists(cat_dir):
            files = sorted(os.listdir(cat_dir))[:6]
            for f in files:
                samples.append({
                    "name": f,
                    "category": "Ship" if category == "ships" else "No Ship",
                    "url": f"/data/sample_dataset/{category}/{f}"
                })

    return jsonify({"samples": samples})


@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory(os.path.join(BASE_DIR, "data"), filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Receives an image (file upload or sample selection), extracts features,
    and runs inference across all available models.
    """
    try:
        if "file" in request.files:
            file = request.files["file"]
            img = Image.open(file.stream).convert("RGB")
        elif request.json and "image_url" in request.json:
            url = request.json["image_url"].lstrip("/")
            full_path = os.path.join(BASE_DIR, url)
            img = Image.open(full_path).convert("RGB")
        elif request.json and "base64" in request.json:
            b64_data = request.json["base64"].split(",")[-1]
            img_bytes = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            return jsonify({"error": "No image provided"}), 400

        # Resize to standard input size 256x256
        img_resized = img.resize((256, 256))
        img_np = np.array(img_resized, dtype=np.uint8)

        # 1. Feature Extraction
        hu = extract_hu_moments(img_np)
        his = extract_color_histogram(img_np)
        ha = extract_haralick_textures(img_np)
        feat_all = extract_features(img_np, feature_set="all")

        # 2. Run Predictions across Traditional Models
        models = get_available_models()
        predictions = {}

        display_names = {
            "RandomForestClassifier": "Random Forest",
            "SVC": "SVM (RBF Kernel)",
            "KNeighborsClassifier": "K-Nearest Neighbors (k=5)",
            "LinearDiscriminantAnalysis": "Linear Discriminant Analysis",
            "GaussianNB": "Gaussian Naive Bayes",
            "MLPClassifier": "Deep Neural Network (MLP)"
        }

        for model_key, pipeline in models.items():
            model = pipeline["model"]
            scaler = pipeline["scaler"]
            
            feat_scaled = scaler.transform(feat_all.reshape(1, -1))
            pred = int(model.predict(feat_scaled)[0])
            
            proba = 0.5
            if hasattr(model, "predict_proba"):
                try:
                    p = model.predict_proba(feat_scaled)[0]
                    proba = float(p[1]) if len(p) > 1 else float(p[0])
                except Exception:
                    proba = 1.0 if pred == 1 else 0.0
            elif hasattr(model, "decision_function"):
                df = model.decision_function(feat_scaled)[0]
                proba = float(1.0 / (1.0 + np.exp(-df)))

            pretty_name = display_names.get(model_key, model_key)
            predictions[pretty_name] = {
                "prediction": "Ship Detected" if proba >= 0.5 else "No Ship",
                "probability": float(round(proba * 100, 2)),
                "is_ship": bool(proba >= 0.5)
            }

        # 3. Best Model Consensus
        consensus_score = float(np.mean([p["probability"] for p in predictions.values()])) if predictions else 50.0
        final_prediction = "Ship Detected" if consensus_score >= 50.0 else "No Ship (Open Water / Coast)"

        # Generate visual maps for features breakdown
        gray_np = rgb_to_gray(img_np)
        hsv_np = rgb_to_hsv_manual(img_np)
        
        # Simple Sobel/gradient edge map using numpy
        gx = np.abs(gray_np[:, 1:].astype(float) - gray_np[:, :-1].astype(float))
        edge_np = np.pad(gx, ((0, 0), (0, 1)), mode='edge')
        edge_np = np.clip((edge_np / (edge_np.max() + 1e-5)) * 255.0, 0, 255).astype(np.uint8)

        def img_to_b64(arr, is_gray=False):
            if is_gray:
                pil_img = Image.fromarray(arr, mode='L')
            else:
                pil_img = Image.fromarray(arr, mode='RGB')
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

        visual_maps = {
            "original": img_to_b64(img_np),
            "grayscale": img_to_b64(gray_np, is_gray=True),
            "hsv": img_to_b64(hsv_np),
            "edges": img_to_b64(edge_np, is_gray=True)
        }

        # Prepare feature payload for charts
        feature_data = {
            "hu_moments": [float(round(float(v), 3)) for v in hu],
            "hu_labels": ["H1 (Inertia)", "H2 (Symmetry)", "H3 (Distortion)", "H4 (Elongation)", "H5 (Skewness)", "H6 (Curvature)", "H7 (Diagonal)"],
            "haralick_textures": {
                "Contrast": float(round(float(ha[0]), 3)),
                "Dissimilarity": float(round(float(ha[2]), 3)),
                "Homogeneity": float(round(float(ha[4]), 3)),
                "Energy": float(round(float(ha[6]), 3)),
                "Correlation": float(round(float(ha[8]), 3)),
                "ASM": float(round(float(ha[10]), 3))
            },
            "color_histogram_sample": [float(round(float(v), 4)) for v in his[:48]],
            "visual_maps": visual_maps
        }

        return jsonify({
            "status": "success",
            "final_prediction": final_prediction,
            "consensus_probability": float(round(consensus_score, 2)),
            "is_ship": bool(consensus_score >= 50.0),
            "model_predictions": predictions,
            "feature_data": feature_data
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    """Returns experiment benchmarks from experiment_results.json."""
    report_file = os.path.join(MODEL_DIR, "experiment_results.json")
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No benchmark file found. Run train_traditional.py first."}), 404


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
    print("Starting Airbus Ship Detection Live Demo Web Server on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
