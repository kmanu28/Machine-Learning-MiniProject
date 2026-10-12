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
- **Important status note:** The committed benchmark numbers were generated earlier on a 125-image demonstration set and must be replaced by the 6,000-image run before submission. They are not presented as Airbus-scale final results.
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
- **Transfer Learning CNN ($\text{TL-CNN}$):** Pretrained ImageNet backbone ($\text{MobileNetV2}$ / $\text{DenseNet169}$) with frozen feature extractor layers and trainable multi-layer classification head.

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

### 4.3 Key Findings

1. Random Forest achieved the best held-out accuracy, reaching 91.6% with color plus Haralick texture features.
2. Raw pixel features performed worse than engineered features for Random Forest, falling from 91.6% with His+Ha to 84.7%.
3. Accuracy alone is misleading for this imbalanced task. Random Forest recall was 54.37% with the all-feature model, so precision, recall, and F1 are reported alongside accuracy.
4. Threshold scanning on the validation set selected 0.35 for Random Forest, improving validation F1 to 0.7898. This threshold was selected before final test reporting.

---

## 5. Conclusions & Future Work
- **Summary:** On the selected Airbus subset, Random Forest with color and Haralick features provided the strongest traditional-ML result. The results demonstrate the value of feature engineering while also showing that class imbalance makes recall and F1 essential metrics.
- **Future Directions:** Compare these results with the Simple CNN and DenseNet-169, tune class weighting, and extend the task toward ship localization or segmentation.