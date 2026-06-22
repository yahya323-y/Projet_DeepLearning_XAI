"""
==========================================
PARTIE III - ETAPE 6: XAI - Visualisation de l'Attention
==========================================

L'attention est naturellement explicable!
Les poids d'attention montrent quels mots source sont
importants pour chaque mot cible genere.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. SIMULATION DE DONNEES D'ATTENTION
# ============================================================

print("=" * 60)
print("VISUALISATION DE L'ATTENTION COMME XAI")
print("=" * 60)

# Phrase source et cible
source = "je suis etudiant en informatique"
target = "i am a computer science student"

source_words = source.split()
target_words = target.split()

print(f"Source: {source}")
print(f"Cible: {target}")


# ============================================================
# 2. MATRICE D'ATTENTION (SIMULEE)
# ============================================================

# Creer une matrice d'attention realiste
# Plus forte attention sur les mots correspondants

np.random.seed(42)
attention_matrix = np.random.rand(len(target_words), len(source_words)) * 0.3

# Renforcer les alignements corrects
alignments = {
    0: [0],      # i -> je
    1: [1],      # am -> suis
    2: [2, 3],   # a student -> etudiant
    3: [4, 5],   # computer -> informatique
    4: [5],      # science -> informatique
    5: [2, 3],   # student -> etudiant
}

for t_idx, s_indices in alignments.items():
    for s_idx in s_indices:
        attention_matrix[t_idx, s_idx] += 0.5

# Normaliser
attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)


# ============================================================
# 3. VISUALISATION HEATMAP
# ============================================================

print("\n" + "=" * 60)
print("1. HEATMAP DE L'ATTENTION")
print("=" * 60)

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(attention_matrix, 
            xticklabels=source_words,
            yticklabels=target_words,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Poids d\'attention'},
            ax=ax)

ax.set_xlabel('Mots Source (Francais)', fontsize=12)
ax.set_ylabel('Mots Cible (Anglais)', fontsize=12)
ax.set_title('Matrice d\'Attention: Source vs Cible', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/attention_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("Heatmap sauvegardee!")


# ============================================================
# 4. VISUALISATION ALIGNEMENT
# ============================================================

print("\n" + "=" * 60)
print("2. ALIGNEMENT SOURCE-CIBLE")
print("=" * 60)

fig, ax = plt.subplots(figsize=(14, 6))

# Dessiner les mots
y_source = 0
y_target = 1

for i, word in enumerate(source_words):
    ax.text(i, y_source, word, ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
            fontsize=11)

for i, word in enumerate(target_words):
    ax.text(i * (len(source_words) / len(target_words)), y_target, word, 
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            fontsize=11)

# Dessiner les lignes d'attention
for t_idx in range(len(target_words)):
    for s_idx in range(len(source_words)):
        if attention_matrix[t_idx, s_idx] > 0.15:  # Seuil
            x1 = s_idx
            x2 = t_idx * (len(source_words) / len(target_words))
            alpha = min(attention_matrix[t_idx, s_idx], 1.0)
            ax.plot([x1, x2], [y_source + 0.1, y_target - 0.1], 
                   'b-', alpha=alpha, linewidth=2*alpha)

ax.set_xlim(-0.5, len(source_words) - 0.5)
ax.set_ylim(-0.3, 1.3)
ax.axis('off')
ax.set_title('Alignement Source-Cible via Attention', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/attention_alignment.png', dpi=150, bbox_inches='tight')
plt.show()

print("Alignement sauvegarde!")


# ============================================================
# 5. EXPLICATION XAI
# ============================================================

print("\n" + "=" * 60)
print("3. EXPLICATION XAI DE L'ATTENTION")
print("=" * 60)

explication = """
L'ATTENTION COMME OUTIL XAI:
==============================

Pourquoi l'attention est explicable:
1. Chaque mot cible "regarde" les mots source avec des poids differents
2. Les poids eleves = mots source importants pour cette traduction
3. Visualisable sous forme de heatmap

Exemple d'interpretation:
- "student" (cible) regarde fortement "etudiant" (source) -> CORRECT
- "computer" (cible) regarde "informatique" (source) -> CORRECT
- "am" (cible) regarde "suis" (source) -> CORRECT

Limites:
- L'attention ne montre pas POURQUOI, seulement QUOI
- Des poids eleves peuvent etre trompeurs
- Ne remplace pas une analyse complete du modele

Avantages par rapport a SHAP/LIME:
- Natif au modele (pas de calcul supplementaire)
- Interpretable par conception
- Visualisation intuitive
"""

print(explication)

with open('../outputs/attention_xai_explanation.txt', 'w') as f:
    f.write(explication)

print("\nExplication XAI sauvegardee!")