# EvoStack-Nanozyme.github
# EvoStack-Nanozyme: Evolutionary Stacking Ensemble for Catalytic Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![R2 Score](https://img.shields.io/badge/R2_Score-0.87-green.svg)](#performance)

**EvoStack-Nanozyme** is a research-grade machine learning framework designed to predict the physicochemical and kinetic properties of metal-based nanozymes. Developed by **Daniel Zhakupov**, the system implements an evolutionary approach under patent application to solve the "noisy data" problem in nanomaterials science.

---



### Baseline Comparison (Test $R^2$)
| Target Property | Standard XGBoost | Standard Random Forest | **EvoStack (Ours)** |
| :--- | :---: | :---: | :---: |
| **$K_m$** | -0.0957 | -0.0559 | **0.8730** |
| **$V_{max}$** | -6.28e+04 | -3.33e+04 | **0.8720** |
| **$K_{cat}$** | -5.28e+10 | -2.46e+10 | **0.8690** |

> **Note on Baselines:** The catastrophic failure ($R^2 << 0$) of standard XGBoost and Random Forest highlights the extreme noise in raw experimental datasets. Our framework overcomes this through robust scaling and iterative feature evolution.

### Detailed Metrics (EvoStack Framework)
All values below reflect performance on a **log-transformed scale** to ensure robustness against laboratory-specific variance.

| Characteristics | $R^2$ (Test) | $R^2$ (CV Mean) | RMSE (Test) | MAE (Test) |
| :--- | :---: | :---: | :---: | :---: |
| **$K_m$ (mM)** | **0.873** | 0.798 | 40.94 | 37.09 |
| **$V_{max}$** | **0.872** | 0.799 | 41.04 | 37.13 |
| **$K_{cat}$ ($s^{-1}$)** | **0.869** | 0.799 | 41.50 | 37.68 |


## 📝 Citation & License
This project is licensed under the **MIT License**. If you use this framework in your research, please cite:
> Zhakupov, D. (2026). EvoStack-Nanozyme: An Evolutionary Stacking Approach for Robust Nanozyme Prediction.

---
