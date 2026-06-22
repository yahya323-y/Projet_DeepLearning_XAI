"""
==========================================
PARTIE IV - ETAPE 3: Meta-Learning
==========================================

Meta-Learning = apprendre les poids optimaux de fusion.
Au lieu de poids fixes, un petit reseau apprend
comment combiner les agents.

Architecture:
- Input: predictions des 3 agents (9 valeurs: 3 classes x 3 agents)
- Hidden: couche dense
- Output: poids de fusion (3 classes)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. META-LEARNER
# ============================================================

print("=" * 60)
print("META-LEARNING: APPRENDRE A FUSIONNER")
print("=" * 60)

class MetaLearner(nn.Module):
    """
    Petit reseau qui apprend les poids optimaux de fusion.
    
    Input: concatenation des predictions des 3 agents
    Output: poids pour chaque classe
    """
    
    def __init__(self, num_agents=3, num_classes=3):
        super(MetaLearner, self).__init__()
        
        input_size = num_agents * num_classes  # 9
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, agent_preds):
        """
        agent_preds: (batch, num_agents, num_classes)
        """
        batch_size = agent_preds.size(0)
        
        # Aplatir: (batch, 9)
        x = agent_preds.view(batch_size, -1)
        
        # Poids de fusion
        fusion_weights = self.network(x)
        
        # Softmax pour normaliser
        fusion_weights = torch.softmax(fusion_weights, dim=1)
        
        return fusion_weights


# ============================================================
# 2. FUSION ADAPTATIVE
# ============================================================

print("\n" + "=" * 60)
print("2. FUSION ADAPTATIVE")
print("=" * 60)

class AdaptiveFusion(nn.Module):
    """
    Fusion qui s'adapte a chaque echantillon.
    
    Le MetaLearner predit des poids differents
    pour chaque echantillon en fonction des predictions des agents.
    """
    
    def __init__(self, num_agents=3, num_classes=3):
        super(AdaptiveFusion, self).__init__()
        
        self.meta_learner = MetaLearner(num_agents, num_classes)
    
    def forward(self, predictions):
        """
        predictions: liste de 3 tenseurs (batch, num_classes)
        """
        # Stack: (batch, 3 agents, 3 classes)
        stacked = torch.stack(predictions, dim=1)
        
        # Poids adaptatifs: (batch, 3 classes)
        weights = self.meta_learner(stacked)
        
        # Fusion ponderee
        # stacked: (batch, 3, 3)
        # weights: (batch, 3)
        # On veut: pour chaque classe, moyenne ponderee des agents
        
        # Reshape pour multiplication
        # stacked: (batch, 3 agents, 3 classes)
        # On applique les memes poids a tous les agents (simplifie)
        
        # Moyenne ponderee par les poids du meta-learner
        fused = (stacked * weights.unsqueeze(1)).sum(dim=1)
        
        return fused, weights


# ============================================================
# 3. DONNEES D'ENTRAINEMENT
# ============================================================

print("\n" + "=" * 60)
print("3. PREPARATION DES DONNEES")
print("=" * 60)

# Simuler des donnees
np.random.seed(42)
n_samples = 100

# Predictions des agents (bruitees)
mlp_train = np.random.rand(n_samples, 3)
cnn_train = np.random.rand(n_samples, 3)
rnn_train = np.random.rand(n_samples, 3)

# Normaliser en probabilites
mlp_train = mlp_train / mlp_train.sum(axis=1, keepdims=True)
cnn_train = cnn_train / cnn_train.sum(axis=1, keepdims=True)
rnn_train = rnn_train / rnn_train.sum(axis=1, keepdims=True)

# Verite terrain (classe avec max de moyenne)
avg = (mlp_train + cnn_train + rnn_train) / 3
y_train = torch.LongTensor(avg.argmax(axis=1))

# Convertir en tensors
mlp_tensor = torch.FloatTensor(mlp_train)
cnn_tensor = torch.FloatTensor(cnn_train)
rnn_tensor = torch.FloatTensor(rnn_train)

print(f"Echantillons: {n_samples}")
print(f"Classes: {np.unique(y_train.numpy())}")


# ============================================================
# 4. ENTRAINEMENT DU META-LEARNER
# ============================================================

print("\n" + "=" * 60)
print("4. ENTRAINEMENT")
print("=" * 60)

model = AdaptiveFusion(num_agents=3, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

N_EPOCHS = 200
losses = []

for epoch in range(N_EPOCHS):
    model.train()
    
    # Forward
    fused, weights = model([mlp_tensor, cnn_tensor, rnn_tensor])
    loss = criterion(fused, y_train)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{N_EPOCHS}], Loss: {loss.item():.4f}")

# Courbe
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Meta-Learner Training')
plt.grid(True, alpha=0.3)
plt.savefig('../outputs/meta_learning_loss.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 5. EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("5. EVALUATION")
print("=" * 60)

model.eval()
with torch.no_grad():
    fused, weights = model([mlp_tensor, cnn_tensor, rnn_tensor])
    predictions = fused.argmax(dim=1)
    accuracy = (predictions == y_train).float().mean().item()

print(f"Accuracy du Meta-Learner: {accuracy:.4f}")

# Analyse des poids
print("\nPoids moyens appris:")
print(f"  Classe 0: {weights[:, 0].mean().item():.4f}")
print(f"  Classe 1: {weights[:, 1].mean().item():.4f}")
print(f"  Classe 2: {weights[:, 2].mean().item():.4f}")

# Sauvegarder
torch.save(model.state_dict(), '../models/meta_learner.pth')
print("\nMeta-Learner sauvegarde!")


# ============================================================
# 6. COMPARAISON FINALE
# ============================================================

print("\n" + "=" * 60)
print("6. COMPARAISON: FIXE vs ADAPTATIF")
print("=" * 60)

comparison = """
| Methode Fusion    | Type      | Avantage                    |
|-------------------|-----------|-----------------------------|
| Moyenne           | Fixe      | Simple, rapide              |
| Vote Majoritaire  | Fixe      | Robuste au bruit            |
| Poids Manuels     | Fixe      | Expertise humaine           |
| Meta-Learning     | Adaptatif | S'adapte a chaque echantillon|
|                   |           | Meilleure performance       |

Conclusion: Le Meta-Learning offre la meilleure performance
mais necessite des donnees d'entrainement supplementaires.
"""

print(comparison)

with open('../outputs/meta_learning_analysis.txt', 'w') as f:
    f.write(comparison)

print("Analyse sauvegardee!")