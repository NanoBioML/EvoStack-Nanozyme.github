# Physics‑Informed Multi‑Output Machine Learning for Nanozyme Kinetic Prediction & Design

![Static Badge](https://img.shields.io/badge/Zhakupov-NanozymeDesigner-blue)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.26434/chemrxiv.15000742v1-blue)](https://doi.org/10.26434/chemrxiv.15000742v1)


## Description
This repository contains the complete end-to-end system for the rational design of nanozymes using a Physics-Informed Multi-Output (PIML) framework. The project integrates thermodynamic and chemical principles (Arrhenius equation, pH-dependent activity zones) into a RegressorChain model using ExtraTreesRegressor to predict core kinetic parameters: Km, kcat, and Vmax.


## Model Performance (R2)
- **Km (mM):** 0.555 (Within 2x factor: 38.9%)
- **kcat (s-1):** 0.319 (Within 2x factor: 16.7%)
- **Vmax (nM/s):** 0.524 (Within 2x factor: 55.6%)
- **Learned Ea:** Peroxidase (1.40 eV), Catalase (0.80 eV)



## Citation
If you utilize this framework, please cite:
> Zhakupov, Daniel. "Physics-Informed Multi-Output Machine Learning for Predicting Nanozyme Kinetic Parameters". ChemRxiv, 2026. DOI: 10.26434/chemrxiv.15000742.v1

## License
Licensed under the MIT License.
