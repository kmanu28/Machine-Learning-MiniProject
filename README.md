# Airbus Ship Detection: Traditional Machine Learning vs. Deep Learning
**Course:** UE24CS352A - Machine Learning Mini-Project  
**Problem Statement:** Serial No. 35 (Airbus Ship Detection Benchmark)  

---

## 📌 Project Overview
Autonomous ship detection from satellite imagery plays a pivotal role in maritime surveillance, port traffic management, disaster response, and prevention of illegal maritime activities. This repository implements and evaluates an end-to-end maritime vision pipeline that benchmarks:
- **5 Traditional Machine Learning Classifiers:** Linear Discriminant Analysis (LDA), K-Nearest Neighbors (KNN), Gaussian Naive Bayes (GNB), Random Forest (RF), and Support Vector Machines (SVM with RBF kernel).
- **Domain Feature Engineering:** 531-dimensional combined feature space comprising **Hu Invariant Moments** (shape), **HSV 3D Color Histograms** (spectral), and **Haralick Textures via GLCM** (high-frequency spatial details).
- **Neural Network Models:** Custom CNN (`SIM-CNN`) trained from scratch and DenseNet-169 transfer CNN (`TL-CNN`). The legacy Deep MLP remains available as an optional baseline but is not part of the final headline comparison.
- **CPU-Friendly Proposed Method:** Hybrid Feature-Fusion Ship Detector (`HFSD`) using compact HOG, spatial HSV statistics, Hu moments, HSV histograms, and Haralick texture features with a calibrated classifier ensemble.
- **Tuned CPU Variant:** `HFSD-Pro` adds multi-scale HOG, local LBP/contrast statistics, HSV quantiles, and validation-only ensemble tuning.
- **Interactive Live Demonstration App:** A glassmorphic web dashboard for real-time aerial patch classification, multi-model consensus, and instant feature extraction visualization.

---

## 📂 Repository Structure

```
mini project/
├── data/
│   ├── sample_dataset/            # Auto-generated satellite image tiles (ships & no_ships)
│   │   ├── ships/                 # Ship vessel satellite patches (256x256)
│   │   └── no_ships/              # Open ocean, clouds, coastline patches (256x256)
│   └── features_all.npz           # Serialized feature extraction matrix
├── src/
│   ├── __init__.py
│   ├── features.py                # Hu moments, HSV histogram, Haralick GLCM textures
│   ├── dataset.py                 # Dataset loader, synthesizer & data augmentations
│   ├── models_traditional.py      # LDA, KNN, NB, RF, SVM + 10-fold CV & threshold scan
│   ├── models_cnn.py              # PyTorch SIM-CNN and TL-CNN architectures
│   ├── models_neural.py           # Zero-dependency Deep MLP classifier
│   ├── train_traditional.py       # Full benchmark runner reproducing Table 1 & CV
│   ├── train_cnn.py               # CNN training loop and loss curve logger
│   └── visualize.py               # Diagnostic plotting (Box plots, ROC/Threshold curves)
├── scripts/
│   ├── prepare_subset.py          # Build the reproducible 6,000-image subset
│   ├── prepare_subset_from_zip.py # Extract only the needed images from the ZIP
│   ├── run_subset_experiment.py   # Traditional ML benchmark
│   ├── run_hfsd_experiment.py     # First CPU-only HFSD method
│   └── run_hfsd_pro_experiment.py # Final tuned CPU-only HFSD-Pro method
├── models/
│   ├── RandomForestClassifier_all.joblib   # Serialized trained models
│   ├── SVC_all.joblib
│   ├── KNeighborsClassifier_all.joblib
│   ├── LinearDiscriminantAnalysis_all.joblib
│   ├── GaussianNB_all.joblib
│   ├── MLPClassifier_all.joblib
│   ├── cv_boxplot.png             # 10-Fold CV accuracy box plot
│   ├── threshold_scanning.png     # Decision threshold optimization curve
│   ├── experiment_results.json    # Traditional ML benchmark metrics
│   ├── hfsd_results.json          # HFSD metrics
│   └── hfsd_pro_results.json      # HFSD-Pro metrics
├── templates/
│   └── index.html                 # Modern glassmorphic Web Demo Dashboard
├── notebooks/
│   └── Airbus_Ship_Detection_Experiments.ipynb  # Interactive experiment notebook
├── report/
│   ├── writeup.md                 # 2-Page project summary write-up (Markdown)
│   ├── writeup.pdf                # 2-Page official PDF report (ReportLab generated)
│   ├── presentation_slides.md     # Complete slide deck for evaluation review
│   └── generate_pdf_writeup.py    # Automated PDF compiler
├── app.py                         # Live Demo Web Application server
└── README.md                      # Documentation
```

---

## ⚡ Quick Start & Setup Instructions

### 1. Install Dependencies
```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn pillow opencv-python scikit-image flask reportlab joblib
```

### 2. Prepare a reproducible smaller subset
If you have a folder containing `ships/` and `no_ships/`, create a 4,000/1,000/1,000 stratified subset:
```bash
python scripts/prepare_subset.py --source /path/to/airbus_images --output data/subset_6000
``` 
The source may be the raw Airbus download containing `train_v2/` and `train_ship_segmentations_v2.csv`; the script derives ship/no-ship labels from the masks. It also accepts pre-labeled `ships/` and `no_ships/` folders. The script removes exact duplicates, uses seed 42, preserves the approximate 16% ship prevalence, and writes a manifest.

Run the fixed-split traditional-ML experiment:
```bash
python scripts/run_subset_experiment.py --data-dir data/subset_6000
```

Run the proposed CPU-only HFSD method. It does not require PyTorch, CUDA, or
a GPU:
```bash
python scripts/run_hfsd_experiment.py --data-dir data/subset_6000
```
This saves `models/hfsd_results.json`, `models/hfsd_pipeline.joblib`, and
`models/hfsd_threshold_scan.png`.

For the stronger tuned CPU variant:
```bash
python scripts/run_hfsd_pro_experiment.py --data-dir data/subset_6000
```
This saves `models/hfsd_pro_results.json` and
`models/hfsd_pro_pipeline.joblib`. The first run extracts and caches 1,259
features per image; later runs reuse the cache.

For the original ZIP download, use the ZIP-aware version so the full archive is not extracted:
```bash
python scripts/prepare_subset_from_zip.py --zip C:\path\to\airbus-ship-detection.zip --output data\subset_6000
```
### 3. Run Complete Training & Benchmarking Pipeline
To run the ablation study across all 8 feature combinations and 10-fold cross-validation:
```bash
python src/train_traditional.py --help
# The current runner can be launched with its configured data directory
python src/train_traditional.py
```

### 4. Generate Diagnostic Visualizations
```bash
python src/visualize.py
```

### 5. Launch Live Interactive Web Demo
Start the local demonstration dashboard:
```bash
python app.py
```
Open your browser at **`http://127.0.0.1:5000`**. You can:
- Drag and drop any satellite image tile.
- Click preset benchmark tiles (ships in open water, ships near coast, clouds over sea).
- Inspect live Hu Moments, Haralick Textures, and HSV Histograms.
- View real-time multi-model ensemble consensus and probability gauges.

---

## 📊 Experimental Results

### Table 1: Held-out test accuracy on the Airbus 6,000-image subset

| Algorithm | His | Ha | Hu | His+Ha | Ha+Hu | Hu+His | All | Raw |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **LDA** | 84.4% | 84.3% | 84.0% | 86.0% | 84.5% | 84.0% | 85.5% | 66.5% |
| **KNN** | 85.1% | 87.1% | 81.0% | 86.9% | 85.3% | 83.7% | 86.5% | 82.8% |
| **Naive Bayes** | 33.7% | 70.6% | 16.3% | 33.8% | 19.7% | 33.9% | 33.9% | 46.5% |
| **Random Forest** | 91.4% | 87.4% | 83.9% | **91.6%** | 88.0% | 91.2% | 91.3% | 84.7% |
| **SVM** | 85.1% | 86.5% | 84.0% | 85.4% | 86.1% | 85.2% | 85.5% | 83.9% |

### 10-Fold Stratified Cross-Validation

- **Random Forest:** 91.97% ± 1.08% accuracy; precision 88.13%, recall 57.81%, F1 69.67%
- **KNN:** 85.72% ± 1.43% accuracy; precision 59.24%, recall 34.53%, F1 43.27%
- **SVM:** 84.85% ± 0.49% accuracy; precision 80.54%, recall 7.97%, F1 14.28%
- **LDA:** 84.10% ± 0.91% accuracy; precision 51.30%, recall 36.09%, F1 41.24%
- **Naive Bayes:** 36.95% ± 1.58% accuracy; precision 19.17%, recall 91.41%, F1 31.68%

Random Forest with His+Ha features produced the best held-out accuracy. The validation threshold scan selected 0.35 and achieved F1 = 0.7898.

### CNN and Proposed-Method Comparison

| Model | Accuracy | Precision | Recall | F1 | Compute |
|:---|---:|---:|---:|---:|:---|
| SIM-CNN | 85.6% | 86.4% | 11.9% | 20.9% | GPU recommended |
| DenseNet-169 / TL-CNN | **92.9%** | 86.8% | 65.6% | 74.7% | Colab GPU; pretrained |
| HFSD-Pro | 92.0% | 73.8% | **77.5%** | **75.6%** | CPU-only; no pretrained weights |

DenseNet-169 has the highest raw accuracy. HFSD-Pro has the best measured F1
and recall balance while running entirely on CPU.

### Proposed HFSD CPU Benchmark

The CPU-only HFSD run on the same fixed split produced the following held-out
test metrics. The threshold was selected on validation data, not on the test
set:

| Method | Accuracy | Precision | Recall | F1 | ROC-AUC |
|:---|---:|---:|---:|---:|---:|
| **HFSD (threshold 0.28)** | 90.4% | 66.8% | **79.4%** | **72.6%** | 92.9% |

At the default threshold of 0.50, HFSD reaches 91.8% accuracy, 65.6% recall,
and 71.9% F1. The lower validation-selected threshold intentionally favors
ship recall, which is more informative than raw accuracy for this imbalanced
dataset.

### HFSD-Pro CPU Benchmark

HFSD-Pro was tuned only on the validation split and evaluated once on the
held-out test split:

| Method | Accuracy | Precision | Recall | F1 | ROC-AUC |
|:---|---:|---:|---:|---:|---:|
| **HFSD-Pro (threshold 0.40)** | **92.0%** | 73.8% | **77.5%** | **75.6%** | 93.9% |

HFSD-Pro has the best measured F1-score among the current models while using
CPU-only feature extraction and inference. DenseNet-169 remains slightly
better on raw accuracy at 92.9%, but its recall is lower at 65.6%.

---

## 📄 Deliverables Summary
1. **Source Code:** Complete modular Python codebase in `src/`, `models/`, and `app.py`.
2. **Colab Notebook:** Run [Airbus_Ship_Detection_Colab_Complete.ipynb](notebooks/Airbus_Ship_Detection_Colab_Complete.ipynb) to train and evaluate both CNNs using the Drive subset.
3. **Two-Page Write-up:** Available in [`report/writeup.pdf`](report/writeup.pdf) and [`report/writeup.md`](report/writeup.md).
4. **Presentation Deck:** Available in [`report/presentation_slides.md`](report/presentation_slides.md). It covers the reference paper, our reproduction, HFSD-Pro, and the side-by-side results.
5. **Live Demonstration:** Interactive Web Dashboard running via `python app.py`.
