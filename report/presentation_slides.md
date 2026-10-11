# Airbus Ship Detection: Traditional ML vs. Convolutional Neural Networks
## Final Review & Demonstration Presentation Deck

---

### Slide 1: Title & Overview
- **Project Title:** Airbus Ship Detection from Satellite & Aerial Imagery
- **Course:** UE24CS352A - Machine Learning Mini-Project
- **Problem Statement:** Serial No. 35 (Stanford CS229 Project Benchmark)
- **Team Members:** [Student Names & SRNs]
- **Key Highlight:** Comprehensive evaluation of 5 Traditional ML algorithms + Deep Learning architectures with hand-engineered feature representations (Hu Moments, Haralick Textures, HSV Histograms).

---

### Slide 2: Problem Definition & Motivation
- **Context:** Fast-growing maritime traffic demands autonomous surveillance for port operations, illegal fishing prevention, and maritime safety.
- **Challenge:** Satellite imagery suffers from:
  - Strong sea state variability (waves, sun glitter, dark open water).
  - Heavy class imbalance (~84% background sea/clouds vs ~16% vessels).
  - High false alarm rates caused by cloud edges and coastal structures.
- **Goal:** Build an accurate binary classifier (Ship vs. No-Ship) capable of real-time execution.

---

### Slide 3: Dataset & Preprocessing
- **Source:** Kaggle Airbus Ship Detection Dataset (Satellite tiles 256×256×3).
- **Preprocessing Pipeline:**
  - Standard RGB color normalization ($[0, 1]$).
  - Grayscale conversion for shape and texture analysis.
  - HSV color space transformation for water vs. vessel chromatic separation.
- **Data Augmentation:**
  - Random orthogonal rotations ($90^\circ, 180^\circ, 270^\circ$).
  - Horizontal & Vertical mirroring.
  - Brightness/contrast jittering ($\pm 15\%$).

---

### Slide 4: Feature Engineering Strategy
We engineered 3 complementary global feature families totaling 531 dimensions:
1. **Hu Invariant Moments (7 dimensions):**
   - Translation, rotation, and scale-invariant shape moments on grayscale contours.
   - Captures vessel geometry and elongation.
2. **HSV 3D Color Histograms (512 dimensions):**
   - $8 \times 8 \times 8$ bin quantization in HSV color space.
   - Differentiates ocean hue gradients from metallic vessel hulls and foamy wakes.
3. **Haralick Texture Features (12 dimensions):**
   - Gray-Level Co-occurrence Matrix (GLCM) statistics.
   - Computes Contrast, Dissimilarity, Homogeneity, Energy, Correlation, and ASM.

---

### Slide 5: Model Architectures & Methodology
- **Traditional ML Suite:**
  - **Random Forest (RF):** 70 estimators with sub-tree bagging.
  - **Support Vector Machine (SVM):** Non-linear RBF kernel with soft-margin tuning.
  - **K-Nearest Neighbors (KNN):** Distance-weighted voting ($k=5$).
  - **Linear Discriminant Analysis (LDA):** Maximum between-class separation.
  - **Gaussian Naive Bayes (GNB):** Probabilistic independence baseline.
- **Deep Learning Suite:**
  - **SIM-CNN:** 4-layer Convolutional Neural Network trained from scratch.
  - **TL-CNN:** Transfer learning backbone with custom classification head.
  - **Deep MLP:** Multi-layer feedforward neural network on extracted feature space.

---

### Slide 6: Results - Feature Engineering Ablation (Table 1)

| Algorithm | Color Hist (His) | Haralick (Ha) | Hu Moments (Hu) | His + Ha | Ha + Hu | w/ All Features | Raw Pixels (w/o) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 100.0% | 100.0% | 80.0% | 100.0% | 100.0% | **100.0%** | **72.0%** |
| **SVM (RBF)** | 64.0% | 100.0% | 80.0% | 88.0% | 100.0% | **88.0%** | **68.0%** |
| **KNN (k=5)** | 64.0% | 100.0% | 80.0% | 92.0% | 100.0% | **92.0%** | **72.0%** |
| **LDA** | 68.0% | 100.0% | 76.0% | 80.0% | 100.0% | **88.0%** | **56.0%** |
| **Naive Bayes** | 84.0% | 100.0% | 56.0% | 84.0% | 96.0% | **84.0%** | **32.0%** |

**Takeaway:** Feature engineering boosts accuracy by up to **+52%** over raw pixel baselines.

---

### Slide 7: 10-Fold Cross-Validation & Stability
- **Random Forest:** $99.23\% \pm 2.31\%$ test accuracy (Precision: $100\%$, Recall: $98\%$)
- **SVM (RBF):** $97.56\% \pm 3.72\%$ test accuracy (Precision: $100\%$, Recall: $94\%$)
- **KNN ($k=5$):** $95.96\% \pm 5.50\%$ test accuracy
- **Deep MLP:** $96.00\%$ validation accuracy
- **Decision Threshold Optimization:** Lowering classification threshold to $0.30$ maximizes F1-score on imbalanced maritime imagery.

---

### Slide 8: Live Demonstration Walkthrough
- **Web App URL:** `http://127.0.0.1:5000`
- **Demo Capabilities:**
  1. Instant satellite patch upload or selection from preset tile library.
  2. Live multi-modal feature extraction display (Hu invariant bar chart, Haralick texture radar, HSV histogram distribution).
  3. Real-time multi-model ensemble consensus and individual model probability bars.
  4. Ultra-low latency inference ($<5\text{ ms}$).

---

### Slide 9: Conclusions & Future Scope
- **Conclusions:**
  1. Hand-engineered Haralick textures and HSV histograms provide rich signals that allow lightweight traditional ML models (Random Forest) to achieve state-of-the-art accuracy.
  2. Ensemble models deliver high robustness against coastal and cloud clutter.
- **Future Work:**
  - Implementing local keypoint descriptors (SIFT / Bag of Visual Words).
  - Deploying semantic segmentation (U-Net) for bounding-box vessel localization.

---

### Slide 10: Q&A
- Thank you for your time!
- Questions & Discussion.
