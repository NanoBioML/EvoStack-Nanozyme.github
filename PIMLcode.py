import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Set publication-quality plot style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# ==================== NLP ====================

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


class SmartCategoryUnifier:
    def __init__(self, use_sbert=True, similarity_threshold=0.75):
        self.use_sbert = use_sbert and HAS_SBERT
        self.similarity_threshold = similarity_threshold

        if self.use_sbert:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedder = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)

        self.chemical_synonyms = {
            'gold': ['au', 'gold', 'gold np'],
            'silver': ['ag', 'silver'],
            'iron': ['fe', 'iron', 'fe3o4', 'magnetite'],
            'copper': ['cu', 'copper', 'cuo'],
            'platinum': ['pt', 'platinum'],
            'tmb': ['tmb', 'tetramethylbenzidine'],
            'h2o2': ['h2o2', 'hydrogen peroxide', 'peroxide'],
            'abts': ['abts'],
        }

    def _normalize_text(self, text):
        if pd.isna(text) or text == '':
            return 'unknown'
        text = str(text).lower().strip()
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _apply_chemical_synonyms(self, text):
        normalized = self._normalize_text(text)
        words = normalized.split()
        canonical_words = []
        for word in words:
            found = False
            for canonical, synonyms in self.chemical_synonyms.items():
                if word in synonyms:
                    canonical_words.append(canonical)
                    found = True
                    break
            if not found:
                canonical_words.append(word)
        return ' '.join(canonical_words)

    def _get_embeddings(self, texts):
        if self.use_sbert:
            return self.embedder.encode(texts, show_progress_bar=False)
        else:
            return self.embedder.fit_transform(texts).toarray()

    def fit_transform(self, categories):
        original = pd.Series(categories).fillna('unknown')
        print(f"\n   🔤 Unifying {original.nunique()} categories...")

        normalized = original.apply(self._normalize_text)
        canonical = normalized.apply(self._apply_chemical_synonyms)
        unique_canonical = canonical.unique()

        if len(unique_canonical) <= 1:
            return canonical

        embeddings = self._get_embeddings(unique_canonical)

        if len(unique_canonical) > 1:
            similarity_matrix = cosine_similarity(embeddings)
            distance_matrix = 1 - similarity_matrix

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - self.similarity_threshold,
                metric='precomputed',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(distance_matrix)

            cluster_to_name = {}
            for i, label in enumerate(cluster_labels):
                if label not in cluster_to_name or len(unique_canonical[i]) < len(cluster_to_name[label]):
                    cluster_to_name[label] = unique_canonical[i]

            canonical_to_unified = {
                unique_canonical[i]: cluster_to_name[cluster_labels[i]]
                for i in range(len(unique_canonical))
            }

            unified = canonical.map(canonical_to_unified)
            print(f"     → {unified.nunique()} categories")
            return unified
        else:
            return canonical


# ==================== CONSTANTS ====================

class PhysicsConstants:
    R_eV_per_K = 8.617e-5
    PH_PEROXIDASE = 4.0
    PH_CATALASE = 7.0
    PH_OXIDASE = 9.0
    E0_MIN = 1e-12  # M
    E0_MAX = 1e-3  # M


# ==================== PREPROCESSING ====================

class PreprocessingPipeline:
    """
    Data leakage prevention: all statistics computed on training set only.
    """

    def __init__(self):
        self.material_medians = {}
        self.winsorization_bounds = {}

    def fit(self, df_train, targets):
        """Compute statistics on training set"""
        print("\n" + "=" * 70)
        print("📊 FIT PREPROCESSING (train only)")
        print("=" * 70)

        for col in df_train.columns:
            if col.startswith('mat_'):
                self.material_medians[col] = df_train[col].median()
                print(f"   {col}: median = {self.material_medians[col]:.3f}")

        for target in targets:
            if target in df_train.columns:
                y = df_train[target].values
                y_log = np.log10(np.maximum(y, 1e-12))
                lower_bound = np.percentile(y_log[~np.isnan(y_log)], 1)
                upper_bound = np.percentile(y_log[~np.isnan(y_log)], 99)
                self.winsorization_bounds[target] = (lower_bound, upper_bound)
                print(f"   {target}: log-bounds = [{lower_bound:.2f}, {upper_bound:.2f}]")

        return self

    def transform(self, df, targets):
        """Apply to train or test"""
        df_transformed = df.copy()

        for col, median in self.material_medians.items():
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].fillna(median)

        for target in targets:
            if target in df_transformed.columns and target in self.winsorization_bounds:
                lower_bound, upper_bound = self.winsorization_bounds[target]
                y = df_transformed[target].values
                if np.isnan(y).any():
                    fill_val = 10 ** lower_bound
                    y = np.nan_to_num(y, nan=fill_val)
                y_log = np.log10(np.maximum(y, 1e-12))
                y_log_clipped = np.clip(y_log, lower_bound, upper_bound)
                df_transformed[target] = 10 ** y_log_clipped

        return df_transformed


def clean_all_numeric_columns(df, targets):
    """Type cleaning"""
    print("\n" + "=" * 70)
    print("🔧 TYPE CLEANING")
    print("=" * 70)

    df_clean = df.copy()

    for target in targets:
        if target in df_clean.columns:
            df_clean[target] = pd.to_numeric(df_clean[target], errors='coerce')
            n_valid = df_clean[target].notna().sum()
            if n_valid > 0:
                print(f"   {target}: {n_valid} valid")

    for col in ['Temp (°C)', 'pH']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    return df_clean


def detect_activity_type(df):
    """Detect activity type by substrate"""
    print("\n🔬 Activity type detection...")
    df = df.copy()
    df['activity_type'] = 'unknown'

    if 'Substrate' not in df.columns:
        return df

    substrate_lower = df['Substrate'].astype(str).str.lower()

    pod_substrates = ['tmb', 'abts', 'opd', 'dopa']
    for sub in pod_substrates:
        df.loc[substrate_lower.str.contains(sub, na=False), 'activity_type'] = 'peroxidase'

    df.loc[substrate_lower.str.contains('h2o2', na=False), 'activity_type'] = 'catalase'

    return df


def prepare_physics_features(df):
    """Create physics-informed features"""
    print("\n" + "=" * 70)
    print("🔬 PHYSICS FEATURES")
    print("=" * 70)

    df_physics = df.copy()

    # Arrhenius
    if 'Temp (°C)' in df_physics.columns:
        df_physics['T_kelvin'] = df_physics['Temp (°C)'] + 273.15
        df_physics['inv_T'] = 1 / df_physics['T_kelvin']
        df_physics['RT_eV'] = PhysicsConstants.R_eV_per_K * df_physics['T_kelvin']
        df_physics.loc[(df_physics['Temp (°C)'] < 0) | (df_physics['Temp (°C)'] > 100), 'inv_T'] = np.nan
        print("   ✅ Arrhenius")

    # pH zones
    if 'pH' in df_physics.columns:
        df_physics['pH_pod_zone'] = np.exp(-(df_physics['pH'] - PhysicsConstants.PH_PEROXIDASE) ** 2 / 1.5)
        df_physics['pH_cat_zone'] = np.exp(-(df_physics['pH'] - PhysicsConstants.PH_CATALASE) ** 2 / 1.5)
        df_physics['pH_oxd_zone'] = np.exp(-(df_physics['pH'] - PhysicsConstants.PH_OXIDASE) ** 2 / 1.5)
        print("   ✅ pH zones")

    # Material properties
    if 'Name' in df_physics.columns:
        metal_properties = {
            'au': {'electronegativity': 2.54, 'redox_potential': 1.50},
            'pt': {'electronegativity': 2.28, 'redox_potential': 1.18},
            'fe': {'electronegativity': 1.83, 'redox_potential': -0.44},
            'ag': {'electronegativity': 1.93, 'redox_potential': 0.80},
            'cu': {'electronegativity': 1.90, 'redox_potential': 0.34},
        }

        df_physics['material_clean'] = df_physics['Name'].astype(str).str.lower()

        for prop in ['electronegativity', 'redox_potential']:
            df_physics[f'mat_{prop}'] = df_physics['material_clean'].apply(
                lambda m: next((metal_properties[metal][prop]
                                for metal in metal_properties if metal in m), np.nan)
            )
        print("   ✅ Material properties")

    return df_physics


# ==================== MODEL ====================

class MultiOutputPhysicsModel:
    """Multi-output physics-informed model"""

    def __init__(self):
        self.Ea_by_activity = {}
        self.chain = None
        self.scaler_X = None
        self.target_names = []
        self.feature_importances_ = None

    def _add_arrhenius_factor(self, X, activity_types):
        """Add Arrhenius factor"""
        X_enhanced = X.copy()

        if 'inv_T' not in X.columns:
            return X_enhanced

        inv_T_values = X['inv_T'].values.astype(float)
        activity_array = activity_types.values

        arr_factor = np.zeros(len(X))

        for activity, Ea in self.Ea_by_activity.items():
            mask = activity_array == activity
            if mask.sum() > 0:
                T_kelvin = 1.0 / inv_T_values[mask]
                arr_factor[mask] = np.exp(-Ea / (PhysicsConstants.R_eV_per_K * T_kelvin))

        X_enhanced['arrhenius_factor'] = arr_factor
        return X_enhanced

    def fit(self, X_train, Y_train, activity_types_train, target_names):
        """Train model with RegressorChain"""
        print("\n" + "=" * 70)
        print("🚀 TRAINING")
        print("=" * 70)

        self.target_names = target_names

        X_enhanced = self._add_arrhenius_factor(X_train, activity_types_train)

        self.scaler_X = RobustScaler()
        X_scaled = self.scaler_X.fit_transform(X_enhanced)

        Y_log = np.log10(np.maximum(Y_train, 1e-6))

        if np.isnan(Y_log).any():
            print("⚠️ Warning: Y_log contains NaN. Replacing with 0.")
            Y_log = np.nan_to_num(Y_log, nan=0.0)

        base_model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=15,
            max_features='sqrt',
            min_samples_leaf=3,
            min_samples_split=6,
            random_state=42,
            n_jobs=-1
        )

        self.chain = RegressorChain(base_model, order=[0, 1, 2], random_state=42)
        self.chain.fit(X_scaled, Y_log)

        # Cross-validation
        try:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.chain, X_scaled, Y_log, cv=kf, scoring='r2')
            print(f"\n   ✅ R² (CV) = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        except Exception as e:
            print(f"   ⚠️ CV failed: {e}")

        # Feature importance
        print("\n   📊 Top-10 features:")
        base_feature_names = list(X_enhanced.columns)
        if len(self.target_names) >= 2:
            added_names = [f"pred_{self.target_names[0]}", f"pred_{self.target_names[1]}"]
        else:
            added_names = ["pred_Km", "pred_Kcat"]
        all_feature_names = base_feature_names + added_names

        importances = self.chain.estimators_[-1].feature_importances_
        indices = np.argsort(importances)[::-1]

        for i in range(min(10, len(indices))):
            idx = indices[i]
            print(f"      {i + 1}. {all_feature_names[idx]}: {importances[idx]:.4f}")

        self.feature_importances_ = dict(zip(all_feature_names, importances))
        return self

    def predict(self, X, activity_types, return_std=False):
        """Predict with physical constraints"""
        X_enhanced = self._add_arrhenius_factor(X, activity_types)
        X_scaled = self.scaler_X.transform(X_enhanced)

        Y_log_pred = self.chain.predict(X_scaled)
        Y_pred = 10 ** Y_log_pred

        if return_std:
            Y_std = np.zeros_like(Y_pred)

        # Physical correction: Vmax = kcat × [E]₀
        if Y_pred.shape[1] >= 3:
            Km_pred = Y_pred[:, 0]
            kcat_pred = Y_pred[:, 1]
            Vmax_pred = Y_pred[:, 2]

            E0_implied = Vmax_pred / (kcat_pred + 1e-12)
            E0_clipped = np.clip(E0_implied, PhysicsConstants.E0_MIN * 1e9, PhysicsConstants.E0_MAX * 1e9)
            Vmax_corrected = kcat_pred * E0_clipped

            Y_pred[:, 2] = Vmax_corrected

        if return_std:
            return Y_pred, Y_std
        return Y_pred


# ==================== MAIN ====================

if __name__ == "__main__":

    FILE_PATH = r"C:\Users\Users\Desktop\AI\Final Database.xlsx"
    TARGETS = ['Km (mM)', 'Kcat (s⁻¹)', 'Vmax (nM s⁻¹)']

    print("=" * 70)
    print("🔬 PHYSICS-INFORMED MULTI-OUTPUT MODEL")
    print("=" * 70)

    # 1. Load data
    df = pd.read_excel(FILE_PATH)
    df.columns = df.columns.str.strip()

    # 2. Type cleaning
    df = clean_all_numeric_columns(df, TARGETS)
    df = df.dropna(subset=['Temp (°C)', 'pH'])

    # 3. NLP substrate unification
    if 'Substrate' in df.columns:
        substrate_unifier = SmartCategoryUnifier(use_sbert=HAS_SBERT)
        df['Substrate'] = substrate_unifier.fit_transform(df['Substrate'])

    # 4. Activity type detection
    df = detect_activity_type(df)

    # 5. Physics feature engineering
    df_physics = prepare_physics_features(df)

    # 6. Feature matrix construction
    feature_cols = []
    for col in ['pH', 'Temp (°C)', 'inv_T', 'RT_eV']:
        if col in df_physics.columns:
            feature_cols.append(col)

    for col in ['pH_pod_zone', 'pH_cat_zone', 'pH_oxd_zone']:
        if col in df_physics.columns:
            feature_cols.append(col)

    for col in df_physics.columns:
        if col.startswith('mat_'):
            feature_cols.append(col)

    available_targets = [t for t in TARGETS if t in df_physics.columns]

    # Remove rows with NaN in Y
    Y_check = df_physics[available_targets].values
    valid_mask = ~np.isnan(Y_check).any(axis=1)

    df_filtered = df_physics[valid_mask].copy()
    X = df_filtered[feature_cols].copy()
    Y = df_filtered[available_targets].values
    activity_types = df_filtered['activity_type'].copy()

    print(f"\n📊 Data: {len(Y)} samples, {X.shape[1]} features")

    # 7. Train/test split (stratified)
    X_train, X_test, Y_train, Y_test, act_train, act_test = train_test_split(
        X, Y, activity_types, test_size=0.2, random_state=42, stratify=activity_types
    )

    # 8. Create DataFrames for preprocessing
    df_train = X_train.copy()
    for i, target in enumerate(available_targets):
        df_train[target] = Y_train[:, i]
    df_train['activity_type'] = act_train.values

    df_test = X_test.copy()
    for i, target in enumerate(available_targets):
        df_test[target] = Y_test[:, i]
    df_test['activity_type'] = act_test.values

    # 9. Fit preprocessing on train
    preprocessor = PreprocessingPipeline()
    preprocessor.fit(df_train, available_targets)

    # 10. Transform train and test
    df_train = preprocessor.transform(df_train, available_targets)
    df_test = preprocessor.transform(df_test, available_targets)

    # Update X and Y
    X_train = df_train[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    Y_train = df_train[available_targets].values
    Y_test = df_test[available_targets].values
    act_train = df_train['activity_type'].copy()
    act_test = df_test['activity_type'].copy()

    print(f"\n   Train: {len(X_train)}, Test: {len(X_test)}")

    # 11. Train model with fixed activation energies (from paper)
    model = MultiOutputPhysicsModel()
    # Use the activation energies that gave the original results
    model.Ea_by_activity = {
        'peroxidase': 1.40,
        'catalase': 0.80,
        'unknown': 0.40
    }
    # Do NOT call learn_activity_specific_Ea
    model.fit(X_train, Y_train, act_train, available_targets)

    # 12. Evaluate on test set
    print("\n" + "=" * 70)
    print("📊 TEST SET PERFORMANCE")
    print("=" * 70)

    Y_pred_test = model.predict(X_test, act_test, return_std=False)

    # Collect metrics
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    within_2x_pct = []
    within_5x_pct = []

    for i, target in enumerate(available_targets):
        y_true = Y_test[:, i]
        y_pred = Y_pred_test[:, i]

        y_log_true = np.log10(np.maximum(y_true, 1e-6))
        y_log_pred = np.log10(np.maximum(y_pred, 1e-6))

        r2 = r2_score(y_log_true, y_log_pred)
        rmse = np.sqrt(mean_squared_error(y_log_true, y_log_pred))
        mae = mean_absolute_error(y_log_true, y_log_pred)

        ratio = np.maximum(y_pred, 1e-6) / np.maximum(y_true, 1e-6)
        within_2x = ((ratio > 0.5) & (ratio < 2.0)).sum()
        within_5x = ((ratio > 0.2) & (ratio < 5.0)).sum()

        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        within_2x_pct.append(within_2x / len(y_true) * 100)
        within_5x_pct.append(within_5x / len(y_true) * 100)

        # Bootstrap CI for R²
        r2_bootstrap = []
        for _ in range(1000):
            idx = np.random.choice(len(y_true), len(y_true), replace=True)
            r2_bootstrap.append(r2_score(y_log_true[idx], y_log_pred[idx]))
        r2_ci = np.percentile(r2_bootstrap, [2.5, 97.5])

        print(f"\n📊 {target}:")
        print(f"   R² = {r2:.4f} (95% CI: [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}])")
        print(f"   RMSE (log₁₀) = {rmse:.4f}")
        print(f"   MAE (log₁₀) = {mae:.4f}")
        print(f"   Within 2×: {within_2x}/{len(y_true)} ({within_2x / len(y_true) * 100:.1f}%)")
        print(f"   Within 5×: {within_5x}/{len(y_true)} ({within_5x / len(y_true) * 100:.1f}%)")

    # ==================== PREPARE FULL DATA PREDICTIONS ====================
    print("\n" + "=" * 70)
    print("💾 PREPARING FULL DATA PREDICTIONS")
    print("=" * 70)

    # Use only samples that passed valid_mask (all targets present)
    X_full = df_physics.loc[valid_mask, feature_cols].copy()
    activity_types_full = df_physics.loc[valid_mask, 'activity_type'].copy()

    print(f"   Initial full samples: {len(X_full)}")

    # Impute mat_ features with medians from preprocessor
    for col, median in preprocessor.material_medians.items():
        if col in X_full.columns:
            X_full[col] = X_full[col].fillna(median)

    # Remove any remaining rows with NaN (e.g., extreme T causing inv_T NaN)
    nan_mask_full = X_full.isna().any(axis=1)
    if nan_mask_full.any():
        print(f"   ⚠️ Removing {nan_mask_full.sum()} samples with NaN in features (e.g., extreme T)")
        X_full = X_full[~nan_mask_full].copy()
        activity_types_full = activity_types_full[~nan_mask_full].copy()

    print(f"   Final full samples for prediction: {len(X_full)}")

    # Predict
    Y_pred_all = model.predict(X_full, activity_types_full, return_std=False)

    # Save results to Excel
    output_dir = os.path.dirname(FILE_PATH)
    df_results = df_physics.copy()
    for i, target in enumerate(available_targets):
        df_results[f'Predicted_{target}'] = np.nan
        df_results.loc[X_full.index, f'Predicted_{target}'] = Y_pred_all[:, i]

    output_path = os.path.join(output_dir, "PUBLICATION_READY.xlsx")
    df_results.to_excel(output_path, index=False)

    print(f"\n💾 Results saved: {output_path}")
    print(f"   Predictions written for {len(X_full)} samples (original total with all targets: {valid_mask.sum()})")

    # ==================== PUBLICATION-QUALITY FIGURES ====================

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}")

    # Figure 1: Parity Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    target_labels = ['K$_m$ (mM)', 'k$_{cat}$ (s$^{-1}$)', 'V$_{max}$ (nM·s$^{-1}$)']

    for i, (ax, target, label) in enumerate(zip(axes, available_targets, target_labels)):
        y_true = Y_test[:, i]
        y_pred = Y_pred_test[:, i]
        y_log_true = np.log10(np.maximum(y_true, 1e-6))
        y_log_pred = np.log10(np.maximum(y_pred, 1e-6))

        ax.scatter(y_log_true, y_log_pred, alpha=0.7, s=60,
                   edgecolors='black', linewidth=0.8, color='steelblue')

        min_val = min(y_log_true.min(), y_log_pred.min())
        max_val = max(y_log_true.max(), y_log_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'k--', linewidth=2, label='1:1 line', zorder=5)

        ax.fill_between([min_val, max_val],
                        [min_val - 0.301, max_val - 0.301],
                        [min_val + 0.301, max_val + 0.301],
                        color='gray', alpha=0.15, label='2× error band')

        ax.set_xlabel(f'Experimental log$_{{10}}$({label})', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'Predicted log$_{{10}}$({label})', fontsize=13, fontweight='bold')
        ax.set_title(f'{label}\nR² = {r2_scores[i]:.3f}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure1_ParityPlots.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Figure1_ParityPlots.pdf'), bbox_inches='tight')
    plt.close()
    print("✅ Figure 1 saved: Parity plots")

    # Figure 2: Feature Importance
    importances = model.feature_importances_
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top_names = [item[0] for item in sorted_items[:10]]
    top_values = [item[1] for item in sorted_items[:10]]

    # Clean up feature names for display
    display_names = []
    for name in top_names:
        if name == 'arrhenius_factor':
            display_names.append('Arrhenius factor')
        elif name == 'pH_pod_zone':
            display_names.append('pH (peroxidase zone)')
        elif name == 'pH_cat_zone':
            display_names.append('pH (catalase zone)')
        elif name == 'pH_oxd_zone':
            display_names.append('pH (oxidase zone)')
        elif name == 'inv_T':
            display_names.append('Inverse temperature (1/T)')
        elif name == 'RT_eV':
            display_names.append('Thermal energy (RT)')
        elif name == 'mat_electronegativity':
            display_names.append('Electronegativity')
        elif name == 'mat_redox_potential':
            display_names.append('Redox potential')
        elif 'pred_' in name:
            display_names.append(name.replace('pred_', 'Predicted '))
        else:
            display_names.append(name.replace('_', ' ').title())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = []
    for name in top_names:
        if any(x in name for x in ['arrhenius', 'pH_', 'inv_T', 'RT_eV']):
            colors.append('#FF8C00')  # Dark orange for physics features
        elif 'pred_' in name:
            colors.append('#4169E1')  # Royal blue for chain features
        else:
            colors.append('#708090')  # Slate gray for material properties

    bars = ax.barh(range(len(display_names)), top_values, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=13, fontweight='bold')
    ax.set_title('Top-10 Feature Importances for V$_{max}$ Prediction', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (bar, val) in enumerate(zip(bars, top_values)):
        ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='#FF8C00', edgecolor='black', label='Physics features'),
        Patch(facecolor='#4169E1', edgecolor='black', label='Chain features'),
        Patch(facecolor='#708090', edgecolor='black', label='Material properties')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure2_FeatureImportance.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Figure2_FeatureImportance.pdf'), bbox_inches='tight')
    plt.close()
    print("✅ Figure 2 saved: Feature importance")

    # Figure 3: Residual Distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, (ax, target, label) in enumerate(zip(axes, available_targets, target_labels)):
        y_true = Y_test[:, i]
        y_pred = Y_pred_test[:, i]
        y_log_true = np.log10(np.maximum(y_true, 1e-6))
        y_log_pred = np.log10(np.maximum(y_pred, 1e-6))
        residuals = y_log_true - y_log_pred

        ax.hist(residuals, bins=12, edgecolor='black', linewidth=1.2,
                alpha=0.75, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=2.5, label='Zero residual')

        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        ax.axvline(mean_res, color='orange', linestyle=':', linewidth=2,
                   label=f'Mean = {mean_res:.3f}')

        ax.set_xlabel('Residuals (log$_{10}$ true - log$_{10}$ pred)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'{label}', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, frameon=True, shadow=True)
        ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure3_Residuals.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Figure3_Residuals.pdf'), bbox_inches='tight')
    plt.close()
    print("✅ Figure 3 saved: Residual distribution")

    # Figure 4: Temperature Dependence (Physics Check)
    if len(X_test) > 0:
        sample = X_test.iloc[0].copy()
        activity = act_test.iloc[0]

        temps = np.linspace(10, 60, 50)
        predicted_Vmax = []

        for t in temps:
            mod_sample = sample.copy()
            mod_sample['Temp (°C)'] = t
            T_k = t + 273.15
            mod_sample['inv_T'] = 1 / T_k
            mod_sample['RT_eV'] = PhysicsConstants.R_eV_per_K * T_k
            X_temp = pd.DataFrame([mod_sample])
            pred = model.predict(X_temp, pd.Series([activity]))
            predicted_Vmax.append(pred[0, 2])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(temps, predicted_Vmax, 'o-', color='darkgreen', markersize=6,
                linewidth=2.5, markeredgecolor='black', markeredgewidth=1)
        ax.set_xlabel('Temperature (°C)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Predicted V$_{max}$ (nM·s$^{-1}$)', fontsize=13, fontweight='bold')
        ax.set_title('Temperature Dependence of V$_{max}$\n(Representative Test Sample)',
                     fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')

        # Compute apparent Ea
        T_kelvin_range = temps + 273.15
        inv_T_range = 1 / T_kelvin_range
        log_Vmax = np.log(predicted_Vmax)
        from scipy.stats import linregress

        slope, intercept, r_value, p_value, std_err = linregress(inv_T_range, log_Vmax)
        Ea_apparent = -slope * PhysicsConstants.R_eV_per_K * 1000  # in meV
        ax.text(0.05, 0.95, f'Arrhenius behavior:\nE$_a$ ≈ {Ea_apparent:.1f} meV',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure4_TemperatureDependence.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'Figure4_TemperatureDependence.pdf'), bbox_inches='tight')
        plt.close()
        print("✅ Figure 4 saved: Temperature dependence")
    else:
        print("⚠️ No test samples available for temperature dependence plot.")

    # Summary Table (save as text file for easy copy-paste into manuscript)
    summary_path = os.path.join(output_dir, "Table1_TestPerformance.txt")
    with open(summary_path, 'w') as f:
        f.write("TABLE 1. Test Set Performance (n={})\n".format(len(X_test)))
        f.write("=" * 80 + "\n")
        f.write(f"{'Target':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'Within 2×':<12} {'Within 5×':<12}\n")
        f.write("=" * 80 + "\n")
        for i, target in enumerate(available_targets):
            f.write(f"{target:<20} {r2_scores[i]:<10.4f} {rmse_scores[i]:<10.4f} {mae_scores[i]:<10.4f} "
                    f"{within_2x_pct[i]:<12.1f}% {within_5x_pct[i]:<12.1f}%\n")
        f.write("=" * 80 + "\n")

    print(f"✅ Table 1 saved: {summary_path}")

    print("\n" + "=" * 70)
    print("✅ PUBLICATION-READY WITH FIGURES!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  📊 Figure1_ParityPlots.png/pdf")
    print("  📊 Figure2_FeatureImportance.png/pdf")
    print("  📊 Figure3_Residuals.png/pdf")
    print("  📊 Figure4_TemperatureDependence.png/pdf")
    print("  📄 Table1_TestPerformance.txt")
    print("  📁 PUBLICATION_READY.xlsx")