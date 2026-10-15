# Airbus Ship Detection
## Reference reproduction, independent benchmark, and HFSD-Pro CPU method

### Slide 1: Project Overview

- **Problem:** classify 256 × 256 Airbus satellite tiles as Ship or No Ship.
- **Reference:** Stanford CS229 Airbus ship-detection project.
- **Our experiment:** reproduce the paper's traditional ML and CNN comparison on a reproducible 6,000-image subset.
- **Our contribution:** HFSD-Pro, a CPU-only feature-fusion detector designed for the same task.

**Central question:** what do we gain from engineered features, CNNs, and our CPU method under the same held-out evaluation?

---

### Slide 2: Why Ship Detection Is Difficult

- Ships occupy a small part of many tiles.
- Ocean, wakes, clouds, coastlines, and glare create similar visual patterns.
- The subset contains approximately 16% ship images and 84% no-ship images.
- Accuracy alone can hide missed ships, so we report precision, recall, F1, specificity, and ROC-AUC.

**Task:** maximize useful ship detection while keeping false alarms manageable.

---

### Slide 3: Dataset and Experimental Protocol

| Reference paper | Our project |
|:---|:---|
| 15,000 images: 7,000 train, 3,000 development, 5,000 test | 6,000 images: 4,000 train, 1,000 validation, 1,000 test |
| Paper-scale experiment | CPU/GPU-friendly reproducible subset |
| Reference benchmark | Independent held-out measurements |

- Images are resized to 256 × 256 × 3.
- The train/validation/test split is fixed with ship prevalence preserved.
- The test set remains untouched during feature/model/threshold selection.

---

### Slide 4: Reference Feature Engineering

The reference traditional pipeline uses 531 global features:

1. **HSV histogram, 512 dimensions** — global color distribution of water, hulls, wakes, and clouds.
2. **Haralick GLCM texture, 12 dimensions** — contrast, homogeneity, energy, correlation, and related texture statistics.
3. **Hu moments, 7 dimensions** — global shape and intensity geometry.

The paper compares color, texture, shape, pairwise combinations, all features, and raw pixels.

---

### Slide 5: Models Recreated from the Paper

- **LDA** — linear class separation.
- **KNN** — distance-weighted nearest-neighbor voting.
- **Gaussian Naive Bayes** — probabilistic baseline.
- **Random Forest** — nonlinear tree ensemble.
- **SVM** — RBF-kernel classifier.
- **SIM-CNN** — four convolution/pooling stages trained from scratch.
- **TL-CNN / DenseNet-169** — transfer-learning CNN with a classification head.

The CNN names are project labels: TL-CNN refers to the DenseNet-169 transfer model, while SIM-CNN refers to the paper-style simple CNN.

---

### Slide 6: Our HFSD-Pro Method

HFSD-Pro extends the reference feature representation without using CUDA or pretrained weights.

**Feature pipeline**

`HSV + Haralick + Hu` → `fine HOG edges` → `local LBP/contrast statistics` → `HSV quantiles` → **1,259 fused features**

**Classifier pipeline**

1. Train HistGradientBoosting and balanced Random Forest candidates.
2. Select configurations and blend weights using validation data only.
3. Tune one decision threshold for F1-score on validation data.
4. Evaluate once on the held-out test set.

This is our original contribution: a CPU-friendly detector that combines global appearance with local edge and texture evidence.

---

### Slide 7: Traditional ML Results on Our 6,000-Image Subset

| Feature set | LDA | KNN | NB | RF | SVM |
|:---|---:|---:|---:|---:|---:|
| His | 84.4% | 85.1% | 33.7% | 91.4% | 85.1% |
| Ha | 84.3% | 87.1% | 70.6% | 87.4% | 86.5% |
| Hu | 84.0% | 81.0% | 16.3% | 83.9% | 84.0% |
| His + Ha | 86.0% | 86.9% | 33.8% | **91.6%** | 85.4% |
| All | 85.5% | 86.5% | 33.9% | 91.3% | 85.5% |
| Raw pixels | 66.5% | 82.8% | 46.5% | 84.7% | 83.9% |

**Finding:** color plus texture is the strongest traditional RF representation in this subset.

---

### Slide 8: CNN and HFSD-Pro Results

| Model | Accuracy | Precision | Recall | F1 | Hardware/method |
|:---|---:|---:|---:|---:|:---|
| SIM-CNN | 85.6% | 86.4% | 11.9% | 20.9% | From scratch |
| DenseNet-169 / TL-CNN | **92.9%** | 86.8% | 65.6% | 74.7% | Pretrained GPU model |
| HFSD-Pro | 92.0% | 73.8% | **77.5%** | **75.6%** | CPU-only |

**Interpretation:** DenseNet-169 has the highest raw accuracy. HFSD-Pro has the best measured F1/recall balance and does not require CUDA or pretrained weights.

---

### Slide 9: Reference Benchmarks vs. Our Results

| Technique | Reference paper | Our experiment | What it means |
|:---|---:|---:|:---|
| Random Forest | approximately 93% accuracy | 91.6% accuracy | Smaller subset gives a slightly lower benchmark |
| Simple CNN | approximately 94% accuracy | 85.6% accuracy, 20.9% F1 | Our imbalance exposes missed ships |
| DenseNet-169 | Transfer-learning baseline | 92.9% accuracy, 74.7% F1 | Best raw accuracy in our run |
| HFSD-Pro | Not in reference paper | 92.0% accuracy, 75.6% F1 | New CPU-only method; best measured F1 |

These numbers are not expected to match exactly because the dataset size, split, and training environment differ.

---

### Slide 10: Thresholds and Error Analysis

- The default 0.50 threshold prioritizes conservative positive predictions.
- HFSD-Pro selects threshold **0.40** on validation data.
- At that threshold, HFSD-Pro detects **124 of 160 ships** in the held-out test set.
- It produces 44 false positives and 36 missed ships.
- Threshold selection makes the recall/precision trade-off explicit instead of hiding it behind accuracy.

---

### Slide 11: Demo and Reproducibility

**CPU traditional/HFSD-Pro run**

```bash
python scripts/run_hfsd_pro_experiment.py --data-dir data/subset_6000
```

**CNN run**

Run `notebooks/Airbus_Ship_Detection_Colab_Complete.ipynb` on the fixed Drive subset.

**Outputs**

- `models/experiment_results.json`
- `models/hfsd_pro_results.json`
- `models/hfsd_pro_pipeline.joblib`
- `cnn results/cnn_results.json` locally after downloading Colab artifacts
- `report/writeup.pdf`

---

### Slide 12: Conclusion

- The reference pipeline establishes strong baselines with engineered features and CNNs.
- Our independent 6,000-image run confirms the same broad pattern: engineered features matter, and class imbalance changes the interpretation of accuracy.
- DenseNet-169 gives the highest raw accuracy in our experiment.
- **HFSD-Pro is our original CPU-only method and the strongest measured F1/recall option.**
- The next validation step is repeating the comparison on the paper's full 15,000-image split.

---

### Slide 13: Questions

**Airbus Ship Detection**

Reference reproduction, independent benchmark, and HFSD-Pro CPU method.
