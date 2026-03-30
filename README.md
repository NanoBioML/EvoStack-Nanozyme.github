# NanozymeDesigner: Physics-Informed Multi-Output ML for Enzyme-Mimetic Nanomaterials

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.26434/chemrxiv.15000742/v1-blue.svg)](https://doi.org/10.26434/chemrxiv.15000742/v1)

This repository implements a Physics-Informed Machine Learning (PIML) framework for the rational design and kinetic prediction of nanozymes. By integrating thermodynamic constraints directly into an ensemble learning pipeline, the system predicts Michaelis-Menten parameters (Km, kcat, Vmax) with high physical consistency.

## Scientific Core

The framework addresses the "black box" nature of traditional ML in catalysis by embedding domain-specific constraints:

- Thermodynamic Featurization: Integration of Arrhenius-derived activation energies (Ea) tailored for specific catalytic activities (POD, CAT, OX).
- pH-Dependent Activity Mapping: Implementation of Gaussian-weighted pH zones based on experimental optima (4.0, 7.0, 9.0).
- Constrained Multi-Output Architecture: A RegressorChain utilizing ExtraTreesRegressor (300 estimators) that enforces the Vmax = kcat * [E]0 relationship.
- In Silico Structural Construction: Automated generation of FCC-lattice nanocrystals (~55 atoms) with stochastic surface defects and substrate adsorption modeling (XYZ export).

## Project Structure

- trainer.py: Preprocessing pipeline & model training (Scikit-learn)
- designer.py: NanozymeDesigner inference engine & 3D constructor
- engine_v1.joblib: Pre-trained ensemble weights
- data/NanozymeDB_curated.csv: Standardized dataset (n=86)

## Quick Start (Usage Example)

```python
from designer import NanozymeDesigner

# Initialize the PIML Designer
designer = NanozymeDesigner(model_path="engine_v1.joblib")

# Target: Physiological Catalase Mimic
result = designer.design_nanozyme(
    substrate_smiles="OO", 
    ph=7.4, 
    temperature=37.0, 
    target_activity="CAT"
)

print(f"Top Candidate: {result['champion_element']} | Predicted Vmax: {result['vmax']} nMs-1")
Model Performance
Validated on a curated subset of NanozymeDB, the model demonstrates high robustness on low-N datasets through ensemble variance reduction:

Km (mM): R2 (Log-scale) = 0.555 | Accuracy (Within 2x) = 38.9%

kcat (s-1): R2 (Log-scale) = 0.319 | Accuracy (Within 2x) = 16.7%

Vmax (nM/s): R2 (Log-scale) = 0.524 | Accuracy (Within 2x) = 55.6%

Key finding: Feature importance analysis identifies the Arrhenius factor (0.228) and Electronic descriptors (Electronegativity, d-electrons) as primary drivers of catalytic efficiency.

Citation
If you utilize this framework in your research, please cite the original work:
Zhakupov, D. (2026). Physics-Informed Multi-Output Machine Learning for Predicting Nanozyme Kinetic Parameters. ChemRxiv. DOI: 10.26434/chemrxiv.15000742/v1

License
This project is licensed under the MIT License.
