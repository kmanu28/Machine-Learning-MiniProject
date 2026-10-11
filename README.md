# Airbus Ship Detection: Traditional Machine Learning vs. Deep Learning
**Course:** UE24CS352A - Machine Learning Mini-Project  
**Problem Statement:** Serial No. 35 (Airbus Ship Detection Benchmark)  

---

## 📌 Project Overview
Autonomous ship detection from satellite imagery plays a pivotal role in maritime surveillance, port traffic management, disaster response, and prevention of illegal maritime activities. This repository implements and evaluates an end-to-end maritime vision pipeline that benchmarks:
- **5 Traditional Machine Learning Classifiers:** Linear Discriminant Analysis (LDA), K-Nearest Neighbors (KNN), Gaussian Naive Bayes (GNB), Random Forest (RF), and Support Vector Machines (SVM with RBF kernel).
- **Domain Feature Engineering:** 531-dimensional combined feature space comprising **Hu Invariant Moments** (shape), **HSV 3D Color Histograms** (spectral), and **Haralick Textures via GLCM** (high-frequency spatial details).
- **Neural Network Models:** Custom Convolutional Neural Network (`SIM-CNN`), Transfer Learning (`TL-CNN`), and Deep Multi-Layer Perceptron (`Deep MLP`).
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
├── models/
│   ├── RandomForestClassifier_all.joblib   # Serialized trained models
│   ├── SVC_all.joblib
│   ├── KNeighborsClassifier_all.joblib
│   ├── LinearDiscriminantAnalysis_all.joblib
│   ├── GaussianNB_all.joblib
│   ├── MLPClassifier_all.joblib
│   ├── cv_boxplot.png             # 10-Fold CV accuracy box plot
│   ├── threshold_scanning.png     # Decision threshold optimization curve
│   └── experiment_results.json    # Complete benchmark metrics
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

### Table 1: Test Accuracy Comparison across Feature Sets
| Algorithm | Color Hist ($\text{His}$) | Haralick ($\text{Ha}$) | Hu Moments ($\text{Hu}$) | $\text{His} + \text{Ha}$ | $\text{Ha} + \text{Hu}$ | $\text{Hu} + \text{His}$ | w/ All Features | Raw Pixels ($\text{w/o}$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (RF)** | 100.0% | 100.0% | 80.0% | 100.0% | 100.0% | 96.0% | **100.0%** | 72.0% |
| **SVM (RBF)** | 64.0% | 100.0% | 80.0% | 88.0% | 100.0% | 76.0% | **88.0%** | 68.0% |
| **KNN ($k=5$)** | 64.0% | 100.0% | 80.0% | 92.0% | 100.0% | 84.0% | **92.0%** | 72.0% |
| **LDA** | 68.0% | 100.0% | 76.0% | 80.0% | 100.0% | 56.0% | **88.0%** | 56.0% |
| **Naive Bayes** | 84.0% | 100.0% | 56.0% | 84.0% | 96.0% | 84.0% | **84.0%** | 32.0% |

### 10-Fold Stratified Cross-Validation Summary
- **Random Forest:** $99.23\% \pm 2.31\%$ (Precision: $100\%$, Recall: $98.0\%$)
- **Support Vector Machine (RBF):** $97.56\% \pm 3.72\%$ (Precision: $100\%$, Recall: $94.0\%$)
- **K-Nearest Neighbors ($k=5$):** $95.96\% \pm 5.50\%$ (Precision: $98.0\%$, Recall: $92.0\%$)
- **Deep MLP Neural Network:** $96.00\%$ validation accuracy

---

## 📄 Deliverables Summary
1. **Source Code:** Complete modular Python codebase in `src/`, `models/`, and `app.py`.
2. **Two-Page Write-up:** Available in [`report/writeup.pdf`](report/writeup.pdf) and [`report/writeup.md`](report/writeup.md).
3. **Presentation Deck:** Available in [`report/presentation_slides.md`](report/presentation_slides.md).
4. **Live Demonstration:** Interactive Web Dashboard running via `python app.py`.
