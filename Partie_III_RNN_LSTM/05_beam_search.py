"""
==========================================
PARTIE III - ETAPE 5: Beam Search vs Greedy Decoding
==========================================

2 strategies de decodage pour la generation:

1. Greedy Decoding: choisir toujours le mot le plus probable
   - Rapide mais peut etre sous-optimal

2. Beam Search: garder les K meilleures sequences a chaque etape
   - Plus lent mais meilleure qualite
"""

import torch
import numpy as np
import os


# ============================================================
# 1. CHARGER LE MODELE SEQ2SEQ
# ============================================================

print("=" * 60)
print("COMPARAISON: GREEDY vs BEAM SEARCH")
print("=" * 60)

# Recharger les donnees (simplifie pour demonstration)
pairs = [
    ("je suis etudiant", "i am a student"),
    ("tu es professeur", "you are a teacher"),
    ("il est docteur", "he is a doctor"),
]

# Vocabulaires simplifies
fra_words = ['<PAD>', '<SOS>', '<EOS>', '<UNK>', 'je', 'tu', 'il', 'elle', 
             'nous', 'vous', 'ils', 'suis', 'es', 'est', 'sommes', 'etes', 
             'sont', 'etudiant', 'professeur', 'docteur', 'ingenieur']

eng_words = ['<PAD>', '<SOS>', '<EOS>', '<UNK>', 'i', 'you', 'he', 'she', 
             'we', 'they', 'am', 'are', 'is', 'a', 'an', 'student', 
             'teacher', 'doctor', 'engineer']

fra_to_idx = {w: i for i, w in enumerate(fra_words)}
eng_to_idx = {w: i for i, w in enumerate(eng_words)}
idx_to_eng = {i: w for w, i in eng_to_idx.items()}


# ============================================================
# 2. GREEDY DECODING
# ============================================================

print("\n" + "=" * 60)
print("1. GREEDY DECODING")
print("=" * 60)

def greedy_decode(probabilities):
    """
    Choisit toujours le mot avec la probabilite la plus elevee.
    
    Exemple:
    Probabilites: [0.1, 0.6, 0.2, 0.1]
    Greedy: choisit index 1 (prob=0.6)
    """
    return np.argmax(probabilities)

# Simulation
print("\nExemple de decodage greedy:")
probs = np.array([0.1, 0.6, 0.2, 0.1])
print(f"Probabilites: {probs}")
print(f"Choix greedy: index {greedy_decode(probs)} (prob={probs[greedy_decode(probs)]:.2f})")


# ============================================================
# 3. BEAM SEARCH
# ============================================================

print("\n" + "=" * 60)
print("2. BEAM SEARCH")
print("=" * 60)

def beam_search_decode(probabilities, k=3):
    """
    Garde les K meilleures sequences a chaque etape.
    
    Exemple avec K=2:
    Etape 1: sequences = [(mot1, 0.6), (mot2, 0.3)]
    Etape 2: pour chaque sequence, calculer les prochains mots
             garder les 2 meilleures parmi toutes les combinaisons
    """
    # Simplification: retourner les k meilleurs indices
    return np.argsort(probabilities)[-k:][::-1]

# Simulation
print("\nExemple de beam search (K=3):")
probs = np.array([0.1, 0.6, 0.2, 0.05, 0.05])
top_k = beam_search_decode(probs, k=3)
print(f"Probabilites: {probs}")
print(f"Top 3 indices: {top_k}")
print(f"Probabilites correspondantes: {probs[top_k]}")


# ============================================================
# 4. COMPARAISON VISUELLE
# ============================================================

print("\n" + "=" * 60)
print("3. COMPARAISON DES STRATEGIES")
print("=" * 60)

import matplotlib.pyplot as plt

# Simuler une sequence de generation
sequence_length = 5
vocab_size = 10

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Greedy
ax1 = axes[0]
greedy_path = []
for t in range(sequence_length):
    probs = np.random.dirichlet(np.ones(vocab_size))
    choice = greedy_decode(probs)
    greedy_path.append(choice)
    ax1.bar(range(vocab_size), probs, alpha=0.5)
    ax1.bar(choice, probs[choice], color='red', label='Choisi' if t == 0 else '')
ax1.set_title('Greedy Decoding: 1 choix a chaque etape')
ax1.set_xlabel('Index du mot')
ax1.set_ylabel('Probabilite')

# Beam Search
ax2 = axes[1]
beam_paths = []
for t in range(sequence_length):
    probs = np.random.dirichlet(np.ones(vocab_size))
    top_k = beam_search_decode(probs, k=3)
    beam_paths.append(top_k)
    ax2.bar(range(vocab_size), probs, alpha=0.3)
    for i, idx in enumerate(top_k):
        ax2.bar(idx, probs[idx], color=['red', 'orange', 'yellow'][i], 
                alpha=0.7, label=f'Top {i+1}' if t == 0 else '')
ax2.set_title('Beam Search (K=3): 3 choix gardes a chaque etape')
ax2.set_xlabel('Index du mot')
ax2.set_ylabel('Probabilite')

plt.tight_layout()
plt.savefig('../outputs/beam_search_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVisualisation sauvegardee!")


# ============================================================
# 5. TABLEAU COMPARATIF
# ============================================================

print("\n" + "=" * 60)
print("TABLEAU COMPARATIF")
print("=" * 60)

comparison = """
| Critere          | Greedy Decoding | Beam Search (K=3) | Beam Search (K=5) |
|------------------|-----------------|-------------------|-------------------|
| Vitesse          | Rapide          | Moyen             | Lent              |
| Qualite          | Moyenne         | Bonne             | Tres bonne        |
| Memoire          | Faible          | Moyenne           | Elevee            |
| Diversite        | Faible          | Moyenne           | Elevee            |
| Risque boucle    | Eleve           | Moyen             | Faible            |

Recommandation:
- Greedy: pour les applications temps reel
- Beam K=3: bon compromis qualite/vitesse
- Beam K=5: pour la traduction (qualite maximale)
"""

print(comparison)

# Sauvegarder
with open('../outputs/beam_search_analysis.txt', 'w') as f:
    f.write(comparison)

print("\nAnalyse sauvegardee!")