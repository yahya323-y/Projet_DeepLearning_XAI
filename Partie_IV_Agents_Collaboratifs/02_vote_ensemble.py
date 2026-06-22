"""
==========================================
PARTIE IV - ETAPE 2: Vote Ensemble
==========================================

3 methodes de vote:
1. Hard Voting: vote majoritaire (classe la plus frequente)
2. Soft Voting: moyenne des probabilites
3. Weighted Voting: vote pondere par la confiance
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. SIMULATION DES PREDICTIONS DES AGENTS
# ============================================================

print("=" * 60)
print("VOTE ENSEMBLE - 3 METHODES")
print("=" * 60)

# Simuler des predictions pour 10 echantillons, 3 classes
np.random.seed(42)

# Agent MLP: bon en general
mlp_preds = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7],
    [0.9, 0.05, 0.05],
    [0.3, 0.6, 0.1],
    [0.1, 0.3, 0.6],
    [0.7, 0.2, 0.1],
    [0.2, 0.8, 0.0],
    [0.1, 0.1, 0.8],
    [0.6, 0.3, 0.1]
])

# Agent CNN: moins confiant
cnn_preds = np.array([
    [0.5, 0.3, 0.2],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5],
    [0.6, 0.2, 0.2],
    [0.4, 0.4, 0.2],
    [0.2, 0.4, 0.4],
    [0.5, 0.3, 0.2],
    [0.3, 0.5, 0.2],
    [0.2, 0.2, 0.6],
    [0.4, 0.4, 0.2]
])

# Agent RNN: parfois different
rnn_preds = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.5, 0.2],
    [0.2, 0.2, 0.6],
    [0.8, 0.1, 0.1],
    [0.4, 0.5, 0.1],
    [0.1, 0.4, 0.5],
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7],
    [0.5, 0.4, 0.1]
])

print("Predictions simulees pour 10 echantillons")
print(f"MLP shape: {mlp_preds.shape}")
print(f"CNN shape: {cnn_preds.shape}")
print(f"RNN shape: {rnn_preds.shape}")


# ============================================================
# 2. HARD VOTING
# ============================================================

print("\n" + "=" * 60)
print("1. HARD VOTING (Vote Majoritaire)")
print("=" * 60)

def hard_voting(mlp, cnn, rnn):
    """
    Chaque agent vote pour une classe.
    La classe avec le plus de votes gagne.
    """
    # Obtenir les classes predites
    mlp_classes = mlp.argmax(axis=1)
    cnn_classes = cnn.argmax(axis=1)
    rnn_classes = rnn.argmax(axis=1)
    
    # Compter les votes
    votes = np.zeros((len(mlp), 3))
    for i in range(len(mlp)):
        votes[i, mlp_classes[i]] += 1
        votes[i, cnn_classes[i]] += 1
        votes[i, rnn_classes[i]] += 1
    
    # Classe avec le plus de votes
    final = votes.argmax(axis=1)
    
    return final, votes

hard_result, hard_votes = hard_voting(mlp_preds, cnn_preds, rnn_preds)

print("Votes pour chaque echantillon:")
for i in range(5):
    print(f"  Ech {i}: MLP={mlp_preds[i].argmax()}, CNN={cnn_preds[i].argmax()}, "
          f"RNN={rnn_preds[i].argmax()} -> Final={hard_result[i]}")
    print(f"         Votes: {hard_votes[i]}")


# ============================================================
# 3. SOFT VOTING
# ============================================================

# ============================================================
# 3. SOFT VOTING (Suite)
# ============================================================

print("\n" + "=" * 60)
print("2. SOFT VOTING (Moyenne des Probabilites)")
print("=" * 60)

def soft_voting(mlp, cnn, rnn):
    """
    Moyenne des probabilites de tous les agents.
    Plus robuste car tient compte de la confiance.
    """
    avg = (mlp + cnn + rnn) / 3
    final = avg.argmax(axis=1)
    return final, avg

soft_result, soft_probs = soft_voting(mlp_preds, cnn_preds, rnn_preds)

print("Probabilites fusionnees (echantillon 0):")
print(f"  MLP:    {mlp_preds[0]}")
print(f"  CNN:    {cnn_preds[0]}")
print(f"  RNN:    {rnn_preds[0]}")
print(f"  Fusion: {soft_probs[0]}")
print(f"  Classe: {soft_result[0]}")


# ============================================================
# 4. WEIGHTED VOTING
# ============================================================

print("\n" + "=" * 60)
print("3. WEIGHTED VOTING (Vote Pondere)")
print("=" * 60)

def weighted_voting(mlp, cnn, rnn, weights=[0.5, 0.3, 0.2]):
    """
    Vote pondere par des poids fixes.
    MLP = expert principal (poids plus eleve)
    """
    w_mlp, w_cnn, w_rnn = weights
    
    weighted = (w_mlp * mlp + w_cnn * cnn + w_rnn * rnn) / sum(weights)
    final = weighted.argmax(axis=1)
    
    return final, weighted

weighted_result, weighted_probs = weighted_voting(mlp_preds, cnn_preds, rnn_preds)

print(f"Poids: MLP={0.5}, CNN={0.3}, RNN={0.2}")
print(f"Resultat echantillon 0: {weighted_probs[0]}, Classe: {weighted_result[0]}")


# ============================================================
# 5. VISUALISATION COMPARATIVE
# ============================================================

print("\n" + "=" * 60)
print("4. COMPARAISON VISUELLE")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Echantillon a visualiser
sample_idx = 0

# MLP
axes[0, 0].bar(range(3), mlp_preds[sample_idx], color='blue', alpha=0.7)
axes[0, 0].set_title('Agent MLP')
axes[0, 0].set_ylim(0, 1)
axes[0, 0].set_xticks(range(3))
axes[0, 0].set_xticklabels(['Classe 0', 'Classe 1', 'Classe 2'])

# CNN
axes[0, 1].bar(range(3), cnn_preds[sample_idx], color='green', alpha=0.7)
axes[0, 1].set_title('Agent CNN')
axes[0, 1].set_ylim(0, 1)
axes[0, 1].set_xticks(range(3))
axes[0, 1].set_xticklabels(['Classe 0', 'Classe 1', 'Classe 2'])

# RNN
axes[1, 0].bar(range(3), rnn_preds[sample_idx], color='red', alpha=0.7)
axes[1, 0].set_title('Agent RNN')
axes[1, 0].set_ylim(0, 1)
axes[1, 0].set_xticks(range(3))
axes[1, 0].set_xticklabels(['Classe 0', 'Classe 1', 'Classe 2'])

# Fusion (Soft Voting)
axes[1, 1].bar(range(3), soft_probs[sample_idx], color='purple', alpha=0.7)
axes[1, 1].set_title('Fusion (Soft Voting)')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].set_xticks(range(3))
axes[1, 1].set_xticklabels(['Classe 0', 'Classe 1', 'Classe 2'])

plt.suptitle(f'Predictions des Agents - Echantillon {sample_idx}', fontsize=14)
plt.tight_layout()
plt.savefig('../outputs/vote_ensemble_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualisation sauvegardee!")


# ============================================================
# 6. TABLEAU COMPARATIF
# ============================================================

print("\n" + "=" * 60)
print("5. TABLEAU COMPARATIF DES METHODES")
print("=" * 60)

# Verite terrain (simulee)
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

from sklearn.metrics import accuracy_score

acc_mlp = accuracy_score(y_true, mlp_preds.argmax(axis=1))
acc_cnn = accuracy_score(y_true, cnn_preds.argmax(axis=1))
acc_rnn = accuracy_score(y_true, rnn_preds.argmax(axis=1))
acc_hard = accuracy_score(y_true, hard_result)
acc_soft = accuracy_score(y_true, soft_result)
acc_weighted = accuracy_score(y_true, weighted_result)

comparison = {
    'Methode': ['MLP seul', 'CNN seul', 'RNN seul', 'Hard Voting', 'Soft Voting', 'Weighted Voting'],
    'Accuracy': [acc_mlp, acc_cnn, acc_rnn, acc_hard, acc_soft, acc_weighted],
    'Description': [
        'Expert tabulaire',
        'Expert images',
        'Expert sequences',
        'Vote majoritaire',
        'Moyenne probabilites',
        'Poids: 0.5, 0.3, 0.2'
    ]
}

import pandas as pd
df = pd.DataFrame(comparison)
print(df.to_string(index=False))

# Sauvegarder
df.to_csv('../outputs/vote_ensemble_results.csv', index=False)
print("\nResultats sauvegardes!")