"""
==========================================
PARTIE I - ETAPE 5: XAI avec SHAP
==========================================

SHAP = SHapley Additive exPlanations
Permet de comprendre quelles features influencent le plus les predictions.

Rouge = pousse vers la classe predite
Bleu = pousse contre la classe predite
"""

import torch
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn


# ============================================================
# 1. CHARGER MODELE ET DONNEES
# ============================================================

print("=" * 60)
print("XAI - SHAP EXPLANATIONS")
print("=" * 60)

# Charger les donnees en premier (nous avons besoin de n_features pour reconstruire le modele si necessaire)
chemin_donnees = os.path.join('..', 'data', 'donnees_wine.pkl')
from torch.utils.data import Dataset


class WineDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


with open(chemin_donnees, 'rb') as f:
    donnees = pickle.load(f)

X_test = donnees['X_test']
feature_names = donnees['feature_names']
n_features = donnees.get('n_features', X_test.shape[1])
n_classes = donnees.get('n_classes', 3)

print(f"Features: {feature_names}")

# Charger le modele entraine - accepter soit un modele complet, soit un state_dict sauvegarde
chemin_modele_full = os.path.join('..', 'models', 'mlp_custom_complet.pth')
chemin_state = os.path.join('..', 'models', 'best_mlp_custom.pth')

# Prefer loading the state_dict (safer and avoids unpickling custom classes)
if os.path.exists(chemin_state):
    # Definir une petite classe compatible pour charger le state_dict
    class MLP_Custom(nn.Module):
        def __init__(self, n_features, n_classes, hidden_sizes=[64, 32], dropout=0.3, init_type='xavier'):
            super(MLP_Custom, self).__init__()
            self.fc1 = nn.Linear(n_features, hidden_sizes[0])
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)
            self.fc3 = nn.Linear(hidden_sizes[1], n_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            return x

    model = MLP_Custom(n_features, n_classes)
    state = torch.load(chemin_state, map_location='cpu')
    # state might be a state_dict or a full model; handle both
    if isinstance(state, dict) and any(k.startswith('fc') or k.startswith('fc1') for k in state.keys()):
        model.load_state_dict(state)
        print("Loaded state_dict from best_mlp_custom.pth into MLP_Custom")
    else:
        # Unexpected format; try to assign directly
        model = state
        print("Loaded object from best_mlp_custom.pth (unexpected format)")
    model.eval()
elif os.path.exists(chemin_modele_full):
    # Try to load full model but avoid untrusted unpickling; use weights_only=False only if necessary
    try:
        model = torch.load(chemin_modele_full)
        model.eval()
        print("Loaded full model from mlp_custom_complet.pth")
    except Exception as e:
        raise RuntimeError(f"Failed to load full model: {e}")
else:
    raise FileNotFoundError("No model file found. Expected ../models/best_mlp_custom.pth or ../models/mlp_custom_complet.pth")


# ============================================================
# 2. PREPARER POUR SHAP
# ============================================================

# Fonction de prediction pour SHAP
def predict_fn(X):
    """
    Wrapper pour que SHAP comprenne PyTorch
    """
    X_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
    return probs.numpy()


# Echantillon de reference (background)
background = X_test[:100].numpy()

# Creer l'explainer
explainer = shap.KernelExplainer(predict_fn, background)

print("Explainer SHAP cree!")


# ============================================================
# 3. EXPLIQUER DES PREDICTIONS
# ============================================================

# Selectionner quelques echantillons a expliquer
n_samples = 10
X_explain = X_test[:n_samples].numpy()

print(f"\nExplication de {n_samples} predictions...")

shap_values = explainer.shap_values(X_explain, nsamples=100)

print("SHAP values calculees!")

# Debug: print shapes/types to handle different SHAP outputs
try:
    if isinstance(shap_values, list):
        for i, sv in enumerate(shap_values):
            arr = np.array(sv)
            print(f"shap_values[{i}] shape: {arr.shape}")
    else:
        print(f"shap_values type: {type(shap_values)}, shape: {np.array(shap_values).shape}")
except Exception:
    pass


# ============================================================
# 4. VISUALISATIONS SHAP
# ============================================================

# --- 4.1 Summary Plot (Global) ---
print("\nGeneration des visualisations...")

plt.figure(figsize=(12, 8))

# SHAP summary pour la classe 0
# Extract SHAP values for class 0 in a robust way
sv = shap_values
if isinstance(sv, np.ndarray) and sv.ndim == 3:
    # shape: (n_samples, n_features, n_classes)
    shap_vals_class0 = sv[:, :, 0]
elif isinstance(sv, list):
    shap_vals_class0 = np.array(sv[0])
else:
    shap_vals_class0 = np.array(sv)
# Try the built-in SHAP plot first, but provide a robust matplotlib fallback
try:
    shap.summary_plot(shap_vals_class0, X_explain, feature_names=feature_names, show=False)
except Exception:
    # Normalize to (n_samples, n_features)
    arr = np.array(shap_vals_class0)
    if arr.ndim == 2 and arr.shape[0] == X_explain.shape[1] and arr.shape[1] == X_explain.shape[0]:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] != X_explain.shape[1]:
        raise RuntimeError(f"Unexpected SHAP array shape: {arr.shape}")

    # Beeswarm-like fallback plot when arr matches (n_samples, n_features)
    if arr.shape[0] == X_explain.shape[0] and arr.shape[1] == X_explain.shape[1]:
        plt.figure(figsize=(8, 10))
        n_features_plot = arr.shape[1]
        y_positions = np.arange(n_features_plot)
        for j in range(n_features_plot):
            x_vals = arr[:, j]
            y_jitter = (np.random.rand(len(x_vals)) - 0.5) * 0.2
            plt.scatter(x_vals, np.full_like(x_vals, y_positions[j]) + y_jitter,
                        c=X_explain[:, j], cmap='coolwarm', edgecolors='k', s=40, alpha=0.8)
        plt.yticks(y_positions, feature_names)
        plt.xlabel('SHAP value')
        plt.title('SHAP Summary - Classe 0 (fallback)')
        plt.gca().invert_yaxis()
        plt.colorbar(label='Feature value')
        plt.tight_layout()
    else:
        # If arr doesn't match sample-feature shape, try other possibilities:
        # - If arr.shape == (n_features, n_classes) (mean per feature per class), plot bar for class 0
        if arr.ndim == 2 and arr.shape[0] == len(feature_names) and arr.shape[1] == n_classes:
            class0_vals = arr[:, 0]
            plt.figure(figsize=(8, 6))
            idx = np.argsort(np.abs(class0_vals))[::-1]
            plt.barh(np.array(feature_names)[idx], class0_vals[idx], color='steelblue')
            plt.xlabel('SHAP value (class 0)')
            plt.title('SHAP Summary - Classe 0 (per-feature aggregate)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
        else:
            # As a last resort, compute mean absolute SHAP from whatever shape we have
            flat = arr.ravel()
            mean_abs = np.abs(flat).mean()
            plt.figure(figsize=(8, 6))
            plt.barh(feature_names, np.ones(len(feature_names)) * mean_abs, color='grey')
            plt.title('SHAP Summary - fallback (mean |SHAP|)')
            plt.tight_layout()
plt.title('SHAP Summary - Classe 0')
plt.tight_layout()
plt.savefig('../outputs/shap_summary_classe0.png', dpi=150, bbox_inches='tight')
plt.show()


# --- 4.2 Feature Importance (Bar plot) ---
plt.figure(figsize=(10, 6))

# Moyenne absolue des SHAP values (robuste selon format de shap_values)
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    mean_shap = np.abs(shap_values[:, :, 0]).mean(axis=0)
elif isinstance(shap_values, list):
    mean_shap = np.abs(np.array(shap_values[0])).mean(axis=0)
else:
    arr_sv = np.array(shap_values)
    if arr_sv.ndim == 3:
        mean_shap = np.abs(arr_sv[:, :, 0]).mean(axis=0)
    elif arr_sv.ndim == 2 and arr_sv.shape[0] == X_explain.shape[0]:
        mean_shap = np.abs(arr_sv).mean(axis=0)
    else:
        mean_shap = np.abs(arr_sv).mean(axis=0)
importance_df = list(zip(feature_names, mean_shap))
importance_df.sort(key=lambda x: x[1], reverse=True)

names, values = zip(*importance_df)
plt.barh(range(len(names)), values, align='center')
plt.yticks(range(len(names)), names)
plt.xlabel('Importance moyenne (|SHAP|)')
plt.title('Feature Importance - SHAP')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../outputs/shap_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVisualisations SHAP sauvegardees!")
print("\nInterpretation:")
print("- Features en haut = plus importantes")
print("- Rouge = valeur elevee, Bleu = valeur faible")
print("- Largeur = impact sur la prediction")