"""
==========================================
PARTIE I - ETAPE 4: Comparaison des Initialisations
==========================================

3 strategies testees:
1. Gaussienne (Normal) - N(0, 0.01)
2. Constante - Tous les poids = 0.1
3. Xavier (Glorot) - Uniforme optimise pour ReLU

Objectif: Montrer que l'initialisation affecte la convergence.
"""

import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. CLASSE MLP AVEC INITIALISATION VARIABLE
# ============================================================

class MLP_Init(nn.Module):
    def __init__(self, n_features, n_classes, init_type='xavier'):
        super(MLP_Init, self).__init__()
        
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()
        
        self.init_type = init_type
        self._apply_init()
    
    def _apply_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == 'gaussian':
                    # Distribution normale N(0, 0.01)
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    nn.init.normal_(m.bias, mean=0, std=0.01)
                
                elif self.init_type == 'constant':
                    # Tous les poids = 0.1
                    nn.init.constant_(m.weight, 0.1)
                    nn.init.constant_(m.bias, 0.1)
                
                elif self.init_type == 'xavier':
                    # Xavier/Glorot - meilleure pour ReLU
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================
# 2. TESTER LES 3 INITIALISATIONS
# ============================================================

print("=" * 60)
print("COMPARAISON DES INITIALISATIONS")
print("=" * 60)

# Charger les donnees
chemin_donnees = os.path.join('..', 'data', 'donnees_wine.pkl')
with open(chemin_donnees, 'rb') as f:
    donnees = pickle.load(f)

n_features = donnees['n_features']
n_classes = donnees['n_classes']

init_types = ['gaussian', 'constant', 'xavier']
models = {}

for init_type in init_types:
    print(f"\n--- Initialisation: {init_type.upper()} ---")
    model = MLP_Init(n_features, n_classes, init_type=init_type)
    models[init_type] = model
    
    # Afficher quelques statistiques
    weights = model.fc1.weight.data.numpy()
    print(f"Poids fc1 - Moyenne: {weights.mean():.4f}, Ecart-type: {weights.std():.4f}")
    print(f"Poids fc1 - Min: {weights.min():.4f}, Max: {weights.max():.4f}")


# ============================================================
# 3. VISUALISATION DES DISTRIBUTIONS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (init_type, model) in enumerate(models.items()):
    weights = model.fc1.weight.data.numpy().flatten()
    
    axes[idx].hist(weights, bins=50, alpha=0.7, color=['red', 'green', 'blue'][idx])
    axes[idx].set_title(f'{init_type.upper()}\nMoy={weights.mean():.4f}, Std={weights.std():.4f}')
    axes[idx].set_xlabel('Valeur des poids')
    axes[idx].set_ylabel('Frequence')

plt.suptitle('Distribution des Poids selon l\'Initialisation', fontsize=14)
plt.tight_layout()
plt.savefig('../outputs/initialisation_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVisualisation sauvegardee: ../outputs/initialisation_comparison.png")


# ============================================================
# 4. TABLEAU COMPARATIF
# ============================================================

print("\n" + "=" * 60)
print("TABLEAU COMPARATIF")
print("=" * 60)

comparison_data = []
for init_type in init_types:
    model = MLP_Init(n_features, n_classes, init_type=init_type)
    weights = model.fc1.weight.data.numpy().flatten()
    comparison_data.append({
        'Type': init_type.upper(),
        'Moyenne': f"{weights.mean():.6f}",
        'Ecart-type': f"{weights.std():.6f}",
        'Min': f"{weights.min():.6f}",
        'Max': f"{weights.max():.6f}"
    })

import pandas as pd
print(pd.DataFrame(comparison_data).to_string(index=False))

print("\nInterpretation:")
print("- Xavier (Bleu): Distribution equilibree, meilleure pour ReLU")
print("- Gaussienne (Rouge): Distribution normale etroite")
print("- Constante (Vert): Tous les poids identiques! Mauvais")