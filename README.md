# Nanozyme Designer: Physics-Informed Multi-Output Machine Learning for Nanozyme Design

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

This repository contains a complete end-to-end system for **rational design of nanozymes** (nanomaterials with enzyme‑like catalytic activity) using a physics‑informed multi‑output machine learning framework. The system integrates:

- **Thermodynamic and chemical principles** (Arrhenius equation, pH‑dependent activity zones, material properties) into the model architecture.
- **A multi‑output RegressorChain** (`ExtraTreesRegressor`) to predict the three core kinetic parameters – Michaelis constant ($K_m$), turnover number ($k_{cat}$), and maximum velocity ($V_{max}$) – simultaneously, respecting the physical relation $V_{max}=k_{cat}\cdot[E]_0$.
- **A five‑stage designer** that accepts a substrate SMILES, pH, temperature, and target activity, and returns a candidate nanozyme with full kinetic predictions and a realistic 3D structure (FCC lattice with surface defects and an adsorbed substrate molecule).

The model was trained on a curated dataset of 86 complete experimental records from NanozymeDB and achieves $R^2 = 0.555$ (Km), $0.319$ (kcat), and $0.524$ (Vmax) on the test set, with 55.6% of Vmax predictions falling within a factor of two of experimental values.

---

## 🧠 Key Features

- **Physics‑informed feature engineering:**  
  - Arrhenius factor with activity‑specific activation energies (peroxidase, catalase, oxidase).  
  - Gaussian pH zones centered at literature optima (4.0, 7.0, 9.0).  
  - Material properties (electronegativity, redox potential, ionic radius, d‑electrons).  

- **Multi‑output modeling with physical constraints:**  
  - `RegressorChain` order: $K_m \rightarrow k_{cat} \rightarrow V_{max}$  
  - Post‑prediction correction: $V_{max} = k_{cat} \cdot [E]_0$ with enzyme concentration clipping.

- **End‑to‑end nanozyme designer (`NanozymeDesigner` class):**  
  1. **Semantic & Electronic Audit** – NLP embedding (Sentence‑BERT) + electronic properties (simulated Fukui indices, chemical hardness).  
  2. **HSAB & Descriptor Filter** – hard/soft acid‑base principle to shortlist compatible metals.  
  3. **Physics‑Generative Constructor** – builds coordination complexes and 3D clusters (FCC lattice with random surface defects).  
  4. **Multi‑Output Engine** – predicts $K_m$, $k_{cat}$, $V_{max}$ using the trained model (or a fallback simulation).  
  5. **Sabatier Volcano & Active Loop** – selects the champion by minimising adsorption energy, with iterative refinement.

- **3D structure export:**  
  Generates realistic XYZ files of the nanozyme cluster (FCC lattice, ~55 atoms) with the substrate molecule placed above the surface. Visualise with any molecular viewer (PyMOL, VMD, Avogadro).

- **Leakage‑free validation:**  
  All preprocessing statistics (imputation medians, winsorization bounds) are computed **exclusively on the training set**.

---

## 📁 Repository Structure
