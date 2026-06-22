"""
==========================================
PARTIE I - ETAPE 2: MLP avec nn.Sequential
==========================================

nn.Sequential = construction simple d'un reseau par sequence.
Chaque couche passe a la suivante automatiquement.

Objectifs:
- Comprendre nn.Sequential
- Inspecter les parametres
- Tester la propagation avant
"""

import torch
import torch.nn as nn
import pickle
import os


# ============================================================
# 1. CHARGEMENT DES DONNEES
# ============================================================

print("=" * 60)
print("CHARGEMENT DES DONNEES")
print("=" * 60)

chemin_donnees = os.path.join('..', 'data', 'donnees_wine.pkl')

with open(chemin_donnees, 'rb') as f:
    donnees = pickle.load(f)

n_features = donnees['n_features']
n_classes = donnees['n_classes']

print(f"Features: {n_features}, Classes: {n_classes}")


# ============================================================
# 2. DEFINITION DU MLP AVEC NN.SEQUENTIAL
# ============================================================

print("\n" + "=" * 60)
print("2. CREATION MLP (nn.Sequential)")
print("=" * 60)

"""
Architecture:
Input (13) -> Hidden1 (64) -> ReLU -> Dropout -> 
Hidden2 (32) -> ReLU -> Dropout -> Output (3)
"""

mlp_sequential = nn.Sequential(
    # Couche 1: Input -> Hidden
    nn.Linear(n_features, 64),    # 13 -> 64 neurones
    nn.ReLU(),                     # Fonction d'activation
    nn.Dropout(0.3),               # Regularisation (30% de neurones desactives)
    
    # Couche 2: Hidden -> Hidden
    nn.Linear(64, 32),             # 64 -> 32 neurones
    nn.ReLU(),
    nn.Dropout(0.3),
    
    # Couche 3: Hidden -> Output
    nn.Linear(32, n_classes)       # 32 -> 3 classes
)

print(mlp_sequential)


# ============================================================
# 3. INSPECTION DES PARAMETRES
# ============================================================

print("\n" + "=" * 60)
print("3. INSPECTION DES PARAMETRES")
print("=" * 60)

print("\n--- named_parameters() ---")
for name, param in mlp_sequential.named_parameters():
    print(f"{name}: shape={param.shape}, trainable={param.requires_grad}")

print("\n--- state_dict() (poids sauvegardables) ---")
for key in mlp_sequential.state_dict().keys():
    print(f"{key}: {mlp_sequential.state_dict()[key].shape}")


# ============================================================
# 4. TEST DE PROPAGATION AVANT (FORWARD PASS)
# ============================================================

print("\n" + "=" * 60)
print("4. TEST FORWARD PASS")
print("=" * 60)

# Exemple: batch de 4 echantillons
x_test = torch.randn(4, n_features)  # 4 echantillons, 13 features chacun
output = mlp_sequential(x_test)

print(f"Input shape: {x_test.shape}")
print(f"Output shape: {output.shape}")
print(f"Output (logits):\n{output}")

# Softmax = probabilites
probs = torch.softmax(output, dim=1)
print(f"\nProbabilites:\n{probs}")
print(f"Classes predites: {torch.argmax(probs, dim=1)}")


# ============================================================
# 5. SAUVEGARDE DU MODELE
# ============================================================

print("\n" + "=" * 60)
print("5. SAUVEGARDE")
print("=" * 60)

chemin_modele = os.path.join('..', 'models', 'mlp_sequential.pth')
torch.save(mlp_sequential.state_dict(), chemin_modele)

print(f"Modele sauvegarde: {chemin_modele}")
print("\nResume nn.Sequential:")
print("- Simple a utiliser")
print("- Ordre fixe des couches")
print("- Moins flexible qu'une classe personnalisee")