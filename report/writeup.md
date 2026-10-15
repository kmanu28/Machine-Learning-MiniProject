# Airbus Ship Detection: Traditional Machine Learning vs. Convolutional Neural Networks

**Course:** UE24CS352A - Machine Learning  
**Project Type:** Mini-Project Assignment  
**Problem Statement:** Serial No. 35 - Airbus Ship Detection (Satellite & Aerial Imagery Classification)

---

## 1. Problem Statement & Motivation
Maritime monitoring and sea vessel identification are critical for maritime domain awareness, preventing illegal fishing, enforcing environmental regulations, and optimizing global maritime logistics. High-resolution satellite imagery presents severe challenges due to varied sea surface conditions, cloud occlusion, sun glint, coastlines, and harbour infrastructure. 

The objective of this project is to develop and evaluate high-performance classification models capable of identifying whether a given 256×256 RGB satellite image tile contains sea vessels. We systematically compare **Traditional Machine Learning** methods coupled with domain-specific feature engineering against **Deep Learning (Convolutional Neural Networks)**.

---

## 2. Dataset Description & Preprocessing
- **Source:** Kaggle Airbus Ship Detection Challenge / Satellite Aerial Imagery.
- **Experimental subset:** The reproducible target subset is 6,000 images split into 4,000 training, 1,000 validation, and 1,000 test images. The `scripts/prepare_subset.py` utility preserves approximately 16% ship prevalence and removes exact duplicates.
- **Experiment status:** All reported benchmark numbers below come from the reproducible 6,000-image Airbus subset. The original 125-image data was only a local smoke-test dataset and is not used for the final results.
- **Image Resolution:** 256 × 256 pixels, 3-channel RGB.
- **Class Balance:** Approximately 16% ship images and 84% background (open ocean, clouds, coastline, wharf infrastructure), representing a severe real-world class imbalance.
- **Preprocessing Pipeline:**
  - Pixel normalization to range $[0, 1]$ for neural network backbones.
  - Color space conversion: RGB $\rightarrow$ Grayscale for texture and contour extraction; RGB $\rightarrow$ HSV for color distribution quantification.
  - Data Augmentation: Random rotations ($90^\circ, 180^\circ, 270^\circ$), horizontal/vertical flips, and brightness perturbation ($\pm 15\%$) to enhance generalizability.

---

## 3. Methodology & Feature Engineering

### 3.1 Hand-Engineered Global Features for Traditional ML
To evaluate traditional classifiers, three complementary feature representations were engineered:
1. **Hu Moments ($\text{Hu}$ - 7 dimensions):** Log-transformed 2D central moments invariant to translation, scale, and rotation, capturing overall geometric ship contours against uniform water backgrounds.
2. **Color Histograms ($\text{His}$ - 512 dimensions):** 3D histogram computed across Hue, Saturation, and Value ($8 \times 8 \times 8$ quantization bins) capturing differences between deep ocean hue signatures, white ship wakes, and cloud formations.
3. **Haralick Textures ($\text{Ha}$ - 12 dimensions):** Gray-Level Co-occurrence Matrix ($\text{GLCM}$) statistics (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, Angular Second Moment) capturing high-frequency ship edges vs. low-frequency ocean textures.

### 3.2 Machine Learning Algorithms
We implemented five baseline algorithms:
- **Linear Discriminant Analysis (LDA):** Finds linear decision boundaries maximizing between-class scatter.
- **K-Nearest Neighbors (KNN):** Distance-weighted local neighbor voting ($k=5$).
- **Gaussian Naive Bayes (GNB):** Probabilistic conditional feature independence model.
- **Random Forest (RF):** Ensemble of 70 decision trees with sub-feature bagging.
- **Support Vector Machine (SVM):** Radial Basis Function ($\text{RBF}$) kernel with soft-margin optimization.

### 3.3 Deep Learning Architectures
- **Simple CNN ($\text{SIM-CNN}$):** Custom 4-stage convolutional neural network trained from scratch with Batch Normalization, Max Pooling ($2\times 2$), Dropout ($p=0.5$), and Dense classification head (32,768 $\rightarrow$ 128 $\rightarrow$ 1).
- **Transfer Learning CNN ($\text{TL-CNN}$):** Pretrained ImageNet DenseNet-169 backbone with frozen feature extractor layers and a trainable multi-layer classification head.

### 3.4 Proposed CPU-Friendly Method
We also implement a proposed **Hybrid Feature-Fusion Ship Detector (HFSD)** for
environments without CUDA. HFSD combines the existing 531-dimensional HSV,
Haralick, and Hu feature vector with a compact HOG descriptor and 4-by-4
spatial HSV mean/standard-deviation statistics. A weighted probability blend of
HistGradientBoosting, balanced Random Forest, and class-balanced logistic
regression produces the final score. The operating threshold is selected on
the validation split for F1-score and is then fixed for test evaluation. This
keeps the method CPU-friendly while combining spectral, texture, shape, edge,
and spatial information.

On the fixed 6,000-image experiment, HFSD achieved 90.4% test accuracy,
66.8% precision, 79.4% recall, 72.6% F1, and 92.9% ROC-AUC at its validation-
selected threshold of 0.28. At the default 0.50 threshold it achieved 91.8%
accuracy and 71.9% F1. The selected lower threshold favors detection recall,
which is appropriate because ship images are the minority class.

We then tuned HFSD-Pro using multi-scale HOG, local LBP/contrast statistics,
HSV quantiles, and validation-only ensemble selection. HFSD-Pro achieved
92.0% test accuracy, 73.8% precision, 77.5% recall, 75.6% F1, and 93.9%
ROC-AUC at threshold 0.40. It is the strongest CPU-only method in this
experiment and exceeds DenseNet-169 on F1 and recall, while DenseNet-169
retains the highest raw accuracy at 92.9%.

---

## 4. Experimental Results & Discussion

### 4.1 Feature Engineering Ablation Study (Airbus 6,000-image subset)

The experiment used 4,000 training images, 1,000 validation images, and 1,000 held-out test images. The test set contained 160 ships and 840 no-ship images.

| Algorithm | His | Ha | Hu | His+Ha | Ha+Hu | Hu+His | All | Raw |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| LDA | 84.4% | 84.3% | 84.0% | 86.0% | 84.5% | 84.0% | 85.5% | 66.5% |
| KNN | 85.1% | 87.1% | 81.0% | 86.9% | 85.3% | 83.7% | 86.5% | 82.8% |
| Naive Bayes | 33.7% | 70.6% | 16.3% | 33.8% | 19.7% | 33.9% | 33.9% | 46.5% |
| Random Forest | 91.4% | 87.4% | 83.9% | **91.6%** | 88.0% | 91.2% | 91.3% | 84.7% |
| SVM | 85.1% | 86.5% | 84.0% | 85.4% | 86.1% | 85.2% | 85.5% | 83.9% |

### 4.2 10-Fold Stratified Cross-Validation on Training Data

- **Random Forest:** $91.97\% \pm 1.08\%$ (Precision: $88.13\%$, Recall: $57.81\%$, F1: $69.67\%$)
- **KNN:** $85.72\% \pm 1.43\%$ (Precision: $59.24\%$, Recall: $34.53\%$, F1: $43.27\%$)
- **SVM:** $84.85\% \pm 0.49\%$ (Precision: $80.54\%$, Recall: $7.97\%$, F1: $14.28\%$)
- **LDA:** $84.10\% \pm 0.91\%$ (Precision: $51.30\%$, Recall: $36.09\%$, F1: $41.24\%$)
- **Naive Bayes:** $36.95\% \pm 1.58\%$ (Precision: $19.17\%$, Recall: $91.41\%$, F1: $31.68\%$)

### 4.3 Reference Paper vs. Our Implementation

| Aspect | Reference paper technique | Our implementation |
|:---|:---|:---|
| Dataset scale | 15,000-image split: 7,000 train, 3,000 development, 5,000 test | Reproducible 6,000-image split: 4,000 train, 1,000 validation, 1,000 test |
| Traditional representation | HSV histograms, Haralick texture, and Hu moments | Same three feature families, plus HFSD-Pro HOG, local LBP/contrast, and HSV quantiles |
| Traditional models | LDA, KNN, Naive Bayes, Random Forest, and SVM | Same five baselines, plus the HFSD-Pro tuned CPU ensemble |
| Simple CNN | Four convolution/pooling stages trained from scratch | Same four-stage SIM-CNN trained from scratch |
| Transfer CNN | Pretrained DenseNet-169 classification pipeline | DenseNet-169 with the same fixed split and held-out evaluation |
| Imbalance handling | Decision-threshold scanning | Validation-only threshold tuning; test set remains untouched |

### 4.4 Measured Results: Reference vs. Our Experiment

The supplied paper summary reports approximately 93% Random Forest accuracy and
94% Simple CNN accuracy. These are reference benchmarks, not copied results.
The values below are our independent held-out test measurements:

| Technique | Reference paper benchmark | Our test result | Interpretation |
|:---|---:|---:|:---|
| Random Forest | approximately 93% accuracy | 91.6% accuracy | Slightly lower on the smaller subset |
| Simple CNN / SIM-CNN | approximately 94% accuracy | 85.6% accuracy, 20.9% F1 | Accuracy hides poor ship recall on our imbalanced split |
| DenseNet-169 / TL-CNN | Transfer-learning baseline | 92.9% accuracy, 74.7% F1 | Best raw accuracy in our experiment |
| HFSD-Pro | Not present in the paper | 92.0% accuracy, 77.5% recall, 75.6% F1 | Our original CPU-only method; best measured F1 |

The comparison is fair but not numerically identical: our subset is smaller, and
the class distribution and fixed split differ from the paper's 15,000-image
experiment. HFSD-Pro is our contribution rather than a claim that the original
paper used the same feature fusion or model ensemble.

### 4.5 Key Findings

1. DenseNet-169 achieved the highest raw accuracy in our experiment, while HFSD-Pro achieved the highest measured F1-score and recall balance.
2. HFSD-Pro is the strongest CPU-only option because it uses engineered features and classical models instead of GPU-based neural-network training.
3. Raw accuracy alone is misleading for this imbalanced task; SIM-CNN reached 85.6% accuracy but only 20.9% F1 because it missed most ships.
4. All thresholds were selected using validation data before the held-out test results were reported.

---

## 5. Conclusions & Future Work
- **Summary:** DenseNet-169 gave the highest raw accuracy, while HFSD-Pro provided the strongest CPU-only F1/recall trade-off. The comparison shows why engineered features and class-imbalance-aware thresholds matter.
- **Future Directions:** Repeat the comparison on the paper's full split, add confidence calibration, and extend the task toward ship localization or segmentation.
