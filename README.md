# Nanozyme Designer: Physics-Informed Multi-Output Machine Learning for Nanozyme Design

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI]((https://doi.org/10.26434/chemrxiv.15000742/v))]((https://doi.org/10.26434/chemrxiv.15000742/v))

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
.
├── README.md
├── LICENSE
├── requirements.txt
├── trainer.py # Script to train the model on your own data
├── designer.py # Main class NanozymeDesigner
├── zhakupov_engine.joblib # Trained model (generated after running trainer.py)
├── designer_output/ # Results folder (created automatically)
│ └── <substrate>_<timestamp>/
│ ├── substrate.png # 2D visualization of the substrate
│ ├── structure_0_Au.xyz # 3D structure for candidate #0
│ ├── predictions.csv # Table with predicted Km, kcat, Vmax for all candidates
│ ├── sabatier_volcano.png # Volcano plot of the candidates
│ └── champion_Au_core.xyz # 3D structure of the final champion
└── data/
└── nanozyme_dataset.csv # Example dataset (86 samples)

text

---

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DanielZhakupov/Physics-Informed-Nanozyme.git
   cd Physics-Informed-Nanozyme
Create a virtual environment (recommended):

bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install dependencies:

bash
pip install -r requirements.txt
Dependencies include: numpy, pandas, scikit-learn, matplotlib, joblib, sentence-transformers (optional), rdkit (optional, for 2D/3D visualisation).

🧪 Train the Model on Your Own Data
If you have an Excel file with kinetic data (columns: Name, Substrate, pH, Temp (°C), Km (mM), Kcat (s⁻¹), Vmax (nM s⁻¹)), you can train the model:

bash
python trainer.py --data_path "your_data.xlsx"
The script will:

Clean the data, unify substrates, compute physics‑informed features.

Split into train/test (stratified by activity type).

Preprocess (impute, winsorize) only on the training set.

Train MultiOutputPhysicsModel and save it as zhakupov_engine.joblib.

If you already have a trained model, place it in the root folder and the designer will load it automatically.

🎨 Run the Nanozyme Designer
python
from designer import NanozymeDesigner

designer = NanozymeDesigner()

# Example: design a catalase for H2O2
result = designer.design_nanozyme(
    substrate_smiles="OO",          # hydrogen peroxide
    temperature=37.0,               # °C
    ph=7.4,                         # physiological pH
    target_activity="CAT"           # catalase activity
)
The output will be stored in designer_output/H2O2_20250312_142349/ with all files described above.

📊 Model Performance (from the original paper)
Target	R² (log)	Within 2×	Within 5×
Km (mM)	0.555	38.9%	66.7%
kcat (s⁻¹)	0.319	16.7%	27.8%
Vmax (nM s⁻¹)	0.524	55.6%	83.3%
Feature importance: Arrhenius factor (0.228) > pH (0.132) > pH zones (0.115, 0.104, 0.078).

Learned activation energies: peroxidase 1.40 eV, catalase 0.80 eV.

🔧 Customising the Designer
You can modify the element database, add new substrates, or replace the fallback simulation with your own trained model. The NanozymeDesigner class is designed to be modular:

_initialize_element_database() – add or change metal properties.

generate_reactive_nanozyme_xyz() – adapt 3D generation for other substrates.

physics_generative_constructor() – adjust how coordination complexes are built.

📚 Citation
If you use this code or data in your research, please cite our preprint:

text
Zhakupov, D. Physics-Informed Multi-Output Machine Learning for Predicting Nanozyme Kinetic Parameters. ChemRxiv, 2026. DOI: 10.26434/chemrxiv-2026-xxxxx
📝 License
This project is licensed under the MIT License – see the LICENSE file for details.

🙏 Acknowledgements
We thank the creators of NanozymeDB for making their data publicly available. This work was supported by [funding sources, if any].

📬 Contact
For questions or collaborations, please open an issue or contact Daniel Zhakupov at d.a.zhakupov@gmail.com.
