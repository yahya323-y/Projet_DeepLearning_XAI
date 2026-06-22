"""
==========================================
PARTIE I - ETAPE 6: Ablation Study (MLP)
==========================================

Ablation = Retirer des composantes et observer l'impact.

Ici: Retirer chaque feature individuellement et mesurer le drop d'accuracy.
"""

import torch
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. CHARGER MODELE ET DONNEES
# ============================================================

print("=" * 60)
print("ABLATION STUDY - MLP")
print("=" * 60)

# Charger le modele
chemin_modele = os.path.join('..', 'models', 'mlp_custom_complet.pth')
model = torch.load(chemin_modele)

# Charger les donnees
chemin_donnees = os.path.join('..', 'data', 'donnees_wine.pkl')
with open(chemin_donnees, 'rb') as f:
    donnees = pickle.load(f)

X_test = donnees['X_test']
y_test = donnees['y_test']
feature_names = list(donnees['feature_names'])

print(f"Features: {feature_names}")


# ============================================================
# 2. FONCTION D'EVALUATION AVEC ABLATION
# ============================================================

def evaluate_with_ablation(model, X, y, feature_idx_to_remove=None):
    """
    Evalue le modele en retirant une feature specifique.
    Si feature_idx_to_remove=None, evaluation baseline.
    """
    X_modified = X.clone()
    
    if feature_idx_to_remove is not None:
        # Mettre la feature a 0 (comme si elle etait retiree)
        X_modified[:, feature_idx_to_remove] = 0
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_modified)
        _, predicted = torch.max(outputs, 1)
    
    accuracy = accuracy_score(y.numpy(), predicted.numpy())
    return accuracy


# ============================================================
# 3. ABLATION: RETIRER CHAQUE FEATURE
# ============================================================

print("\n--- Ablation par Feature ---")

baseline_acc = evaluate_with_ablation(model, X_test, y_test, None)
print(f"Baseline (toutes features): {baseline_acc:.4f}")

results = []
results.append(('Baseline (all)', baseline_acc, 0.0))

for idx, feat_name in enumerate(feature_names):
    acc = evaluate_with_ablation(model, X_test, y_test, idx)
    drop = baseline_acc - acc
    results.append((feat_name, acc, drop))
    print(f"Sans '{feat_name}': {acc:.4f} (drop: {drop:.4f})")


# ============================================================
# 4. ABLATION MULTIPLE
# ============================================================

print("\n--- Ablation multiple ---")

# Trier par impact (drop)
results_sorted = sorted(results[1:], key=lambda x: x[2], reverse=True)

print("\nFeatures les PLUS importantes (drop eleve):")
for name, acc, drop in results_sorted[:3]:
    print(f"  {name}: drop={drop:.4f}")

print("\nFeatures les MOINS importantes (drop faible):")
for name, acc, drop in results_sorted[-3:]:
    print(f"  {name}: drop={drop:.4f}")

# Retirer les 3 moins importantes
least_important = [feature_names.index(name) for name, _, _ in results_sorted[-3:]]
X_test_ablated = X_test.clone()
X_test_ablated[:, least_important] = 0

model.eval()
with torch.no_grad():
    outputs = model(X_test_ablated)
    _, predicted = torch.max(outputs, 1)
    acc_ablated = accuracy_score(y_test.numpy(), predicted.numpy())

print(f"\nSans 3 features les moins importantes: {acc_ablated:.4f}")
print(f"Drop total: {baseline_acc - acc_ablated:.4f}")


# ============================================================
# 5. VISUALISATION
# ============================================================

names = [r[0] for r in results[1:]]
accs = [r[1] for r in results[1:]]
drops = [r[2] for r in results[1:]]

plt.figure(figsize=(12, 6))

# Bar plot des drops
colors = ['red' if d > 0.05 else 'orange' if d > 0.02 else 'green' for d in drops]
plt.barh(range(len(names)), drops, color=colors)
plt.yticks(range(len(names)), names)
plt.xlabel('Drop d\'Accuracy')
plt.title('Ablation Study: Impact de chaque Feature')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Legende
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', label='Critique (>5%)'),
    Patch(facecolor='orange', label='Important (2-5%)'),
    Patch(facecolor='green', label='Faible (<2%)')
]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('../outputs/ablation_study_mlp.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nAblation study terminee!")
print("Interpretation:")
print("- Rouge: Feature critique (drop > 5%)")
print("- Orange: Feature importante (drop 2-5%)")
print("- Vert: Feature peu importante (drop < 2%)")