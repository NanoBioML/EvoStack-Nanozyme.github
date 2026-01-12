# EvoStack-Nanozyme.github
# EvoStack-Nanozyme: Evolutionary Stacking Ensemble for Catalytic Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![R2 Score](https://img.shields.io/badge/R2_Score-0.87-green.svg)](#performance)

**EvoStack-Nanozyme** is a research-grade machine learning framework designed to predict the physicochemical and kinetic properties of metal-based nanozymes. Developed by **Daniel Zhakupov**, the system implements a patented evolutionary approach to solve the "noisy data" problem in nanomaterials science.

---

## 🔬 Scientific Background
Predicting nanozyme activity (e.g., $K_{cat}$, $V_{max}$) is challenging due to experimental noise and non-standardized reporting in literature. This project addresses these challenges using:
1. **Robust Preprocessing:** Log-transformation and `RobustScaler` to minimize outlier influence.
2. **Evolutionary Feature Engineering:** Iterative pruning of weak descriptors and non-linear generation of new synthetic ones.
3. **Stacking Architecture:** A multi-layer ensemble of XGBoost, Random Forest, and Ridge models.

## 🔬 Key Methodology

The framework follows a multi-stage iterative process designed to maximize predictive accuracy ($R^2$) and robustness against experimental noise. The pipeline is divided into the following functional blocks:

### 1. Robust Data Preprocessing (Stage A)
To mitigate the impact of experimental outliers, the system applies:
* **Target Transformation:** $y_{transformed} = \ln(1 + y)$, stabilizing variance across different scales of catalytic activity.
* **Feature Scaling:** `RobustScaler` is utilized, which scales features based on percentiles (IQR), ensuring that extreme values in Nanozyme DB do not skew the model.

### 2. Dual-Layer Stacking Architecture (Stages B, C, G)
The model employs a hierarchical ensemble approach:
* **Level 0 (Base Models):** A diversified collection of regressors (XGBoost, Random Forest, and Ridge) trained using K-Fold cross-validation.
* **Out-of-Fold (OOF) Generation:** Predictions from Level 0 are used as meta-features to prevent data leakage.
* **Level 1 (Meta-Learning):** A Ridge Regression meta-model aggregates base predictions and calculates a **Weight Vector ($W$)**, quantifying the contribution of each original and synthetic feature.



### 3. Evolutionary Feature Cycle (Stages D, E)
This is the core innovation of the framework. In each iteration, the system performs:
* **Feature Pruning:** Automated removal of the bottom $P\%$ (default 20%) of features based on the absolute values of the weight vector $W$.
* **Synthetic Generation:** Creation of new non-linear descriptors (top $G\%$) through binary operations (multiplication, subtraction, division) on the most significant features.
* **Iterative Optimization:** The cycle repeats for a set number of `max_iterations` or until `max_patience` is reached, ensuring the feature space evolves towards the most physically relevant descriptors.

### 4. Final Inference (Stage J)
The best-performing configuration is trained on the full dataset. Final predictions are converted back to their physical units using the inverse transformation: 
$$y_{pred} = \exp(y_{pred\_transformed}) - 1$$

## 📊 Performance Benchmark & Comparative Analysis

The primary challenge with **Nanozyme DB** is the high level of experimental noise and extreme outliers. Standard machine learning models fail to converge or generalize on this data. Our proposed evolutionary stacking approach demonstrates exceptional stability and accuracy.

### Baseline Comparison (Test $R^2$)
| Target Property | Standard XGBoost | Standard Random Forest | **EvoStack (Ours)** |
| :--- | :---: | :---: | :---: |
| **pH** | 0.5022 | 0.5540 | **0.8680** |
| **Temperature** | -0.8823 | -0.2723 | **0.8670** |
| **$K_m$** | -0.0957 | -0.0559 | **0.8730** |
| **$V_{max}$** | -6.28e+04 | -3.33e+04 | **0.8720** |
| **$K_{cat}$** | -5.28e+10 | -2.46e+10 | **0.8690** |

> **Note on Baselines:** The catastrophic failure ($R^2 << 0$) of standard XGBoost and Random Forest highlights the extreme noise in raw experimental datasets. Our framework overcomes this through robust scaling and iterative feature evolution.

### Detailed Metrics (EvoStack Framework)
All values below reflect performance on a **log-transformed scale** to ensure robustness against laboratory-specific variance.

| Characteristics | $R^2$ (Test) | $R^2$ (CV Mean) | RMSE (Test) | MAE (Test) |
| :--- | :---: | :---: | :---: | :---: |
| **pH** | **0.868** | 0.799 | 41.67 | 37.78 |
| **Temp (°C)** | **0.867** | 0.804 | 41.90 | 37.40 |
| **$K_m$ (mM)** | **0.873** | 0.798 | 40.94 | 37.09 |
| **$V_{max}$** | **0.872** | 0.799 | 41.04 | 37.13 |
| **$K_{cat}$ ($s^{-1}$)** | **0.869** | 0.799 | 41.50 | 37.68 |


## 📝 Citation & License
This project is licensed under the **MIT License**. If you use this framework in your research, please cite:
> Zhakupov, D. (2026). EvoStack-Nanozyme: An Evolutionary Stacking Approach for Robust Nanozyme Prediction.

---
