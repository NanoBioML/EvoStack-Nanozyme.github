# Physics‑Informed Multi‑Output Machine Learning for Nanozyme Kinetic Prediction & Design

![Static Badge](https://img.shields.io/badge/Zhakupov-NanozymeDesigner-blue)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.26434/chemrxiv.15000742v1-blue)](https://doi.org/10.26434/chemrxiv.15000742v1)


## Description
This repository contains the complete end-to-end system for the rational design of nanozymes using a Physics-Informed Multi-Output (PIML) framework. The project integrates thermodynamic and chemical principles (Arrhenius equation, pH-dependent activity zones) into a RegressorChain model using ExtraTreesRegressor to predict core kinetic parameters: Km, kcat, and Vmax.

## Installation & Quick Start
Ensure you have Python 3.11 or higher installed. Follow these steps:

1. git clone [https://github.com/DanielZhakupov/Physics-Informed-Nanozyme.git](https://github.com/DanielZhakupov/Physics-Informed-Nanozyme.git)
2. cd Physics-Informed-Nanozyme
3. python -m venv venv
4. source venv/bin/activate  # For Windows: venv\Scripts\activate
5. pip install -r requirements.txt
6. python trainer.py        # Trains the Multi-Output Physics Model
7. python designer.py       # Launches the end-to-end generative design loop

## Repository Structure
- **trainer.py**: Script to train the PIML model on experimental data.
- **designer.py**: Main engine for semantic audit, metal filtering, and 3D generation.
- **data/nanozyme_dataset.csv**: Curated dataset from NanozymeDB (n=86).
- **docs/ru/**: User documentation (Introduction & ML fundamentals).
- **designer_output/**: Auto-generated results (XYZ structures, Sabatier plots).

## Usage Example

    from designer import NanozymeDesigner

    designer = NanozymeDesigner()
    result = designer.design_nanozyme(
        substrate_smiles="OO",      # hydrogen peroxide
        temperature=37.0,           # Celsius
        ph=7.4,                     # physiological pH
        target_activity="CAT"       # catalase-like activity
    )

    print(f"Champion Candidate: {result['final_champion']['metal']}-based core")


## Model Performance (R2)
- **Km (mM):** 0.555 (Within 2x factor: 38.9%)
- **kcat (s-1):** 0.319 (Within 2x factor: 16.7%)
- **Vmax (nM/s):** 0.524 (Within 2x factor: 55.6%)
- **Learned Ea:** Peroxidase (1.40 eV), Catalase (0.80 eV)

## Conventional Commits Reference
| Type | Description |
|----------|----------|
| build | Build system changes or external dependencies |
| science | Updates to physical constants or thermodynamics |
| feat | New features (descriptors, model architectures) |
| fix | Bug fixes in calculations or code logic |
| docs | Documentation or README updates |
| perf | Performance optimizations for the algorithm |
| style | Code style changes (PEP8 compliance) |

## Citation
If you utilize this framework, please cite:
> Zhakupov, Daniel. "Physics-Informed Multi-Output Machine Learning for Predicting Nanozyme Kinetic Parameters". ChemRxiv, 2026. DOI: 10.26434/chemrxiv.15000742.v1

## License
Licensed under the MIT License.
