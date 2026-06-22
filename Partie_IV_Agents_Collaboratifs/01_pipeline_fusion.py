"""
==========================================
PARTIE IV - ETAPE 1: Pipeline de Fusion
==========================================

Agents Collaboratifs = 3 modeles specialises qui votent:
- Agent MLP: expert en donnees tabulaires
- Agent CNN: expert en images
- Agent RNN: expert en sequences

Cette etape: creer le pipeline qui fusionne leurs predictions.
"""

import torch
import torch.nn as nn
import numpy as np


# ============================================================
# 1. DEFINITION DES AGENTS
# ============================================================

print("=" * 60)
print("AGENTS COLLABORATIFS - PIPELINE DE FUSION")
print("=" * 60)

class Agent_MLP(nn.Module):
    """Agent specialise en donnees tabulaires"""
    def __init__(self, input_size, num_classes):
        super(Agent_MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self(x)
            return torch.softmax(logits, dim=1)


class Agent_CNN(nn.Module):
    """Agent specialise en images"""
    def __init__(self, num_classes):
        super(Agent_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self(x)
            return torch.softmax(logits, dim=1)


class Agent_RNN(nn.Module):
    """Agent specialise en sequences"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(Agent_RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self(x)
            return torch.softmax(logits, dim=1)


# ============================================================
# 2. PIPELINE DE FUSION
# ============================================================

print("\n" + "=" * 60)
print("2. FUSION DES PREDICTIONS")
print("=" * 60)

class FusionPipeline(nn.Module):
    """
    Pipeline qui fusionne les predictions des 3 agents.
    
    3 methodes de fusion:
    1. Moyenne simple (Average)
    2. Vote majoritaire (Voting)
    3. Apprentissage des poids (Meta-Learning)
    """
    
    def __init__(self, num_classes, fusion_type='average'):
        super(FusionPipeline, self).__init__()
        
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        
        # Poids appris pour la fusion (meta-learning)
        if fusion_type == 'meta':
            self.weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, predictions):
        """
        predictions: liste de 3 tenseurs (batch, num_classes)
                     [mlp_probs, cnn_probs, rnn_probs]
        """
        if self.fusion_type == 'average':
            # Moyenne simple
            fused = torch.stack(predictions).mean(dim=0)
        
        elif self.fusion_type == 'voting':
            # Vote majoritaire (hard voting)
            votes = torch.stack([p.argmax(dim=1) for p in predictions])
            # Compter les votes (simplifie)
            fused = predictions[0]  # Par defaut, prendre le premier
        
        elif self.fusion_type == 'meta':
            # Poids appris (softmax pour normaliser)
            weights = torch.softmax(self.weights, dim=0)
            weighted = torch.stack([
                w * p for w, p in zip(weights, predictions)
            ])
            fused = weighted.sum(dim=0)
        
        return fused


# ============================================================
# 3. TEST DU PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("3. TEST DU PIPELINE")
print("=" * 60)

# Creer les agents
agent_mlp = Agent_MLP(input_size=13, num_classes=3)
agent_cnn = Agent_CNN(num_classes=10)
agent_rnn = Agent_RNN(vocab_size=50, embed_size=32, hidden_size=64, num_classes=5)

# Creer le pipeline
pipeline = FusionPipeline(num_classes=3, fusion_type='average')

# Simuler des entrees
batch_size = 4

# Donnees tabulaires (MLP)
x_tab = torch.randn(batch_size, 13)
proba_mlp = agent_mlp.predict_proba(x_tab)

# Donnees images (CNN) - simulees pour avoir meme nombre de classes
x_img = torch.randn(batch_size, 3, 32, 32)
proba_cnn = agent_cnn.predict_proba(x_img)[:, :3]  # Prendre seulement 3 classes

# Donnees sequences (RNN) - simulees
x_seq = torch.randint(0, 50, (batch_size, 20))
proba_rnn = agent_rnn.predict_proba(x_seq)[:, :3]  # Prendre seulement 3 classes

print(f"Prediction MLP:  {proba_mlp[0]}")
print(f"Prediction CNN:  {proba_cnn[0]}")
print(f"Prediction RNN:  {proba_rnn[0]}")

# Fusion
fused_proba = pipeline([proba_mlp, proba_cnn, proba_rnn])
print(f"\nPrediction fusionnee: {fused_proba[0]}")
print(f"Classe finale: {fused_proba[0].argmax().item()}")


# ============================================================
# 4. EXPLICATION DU PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("4. ARCHITECTURE DU SYSTEME")
print("=" * 60)

architecture = """
┌─────────────────────────────────────────────────────────────┐
│              SYSTEME MULTI-AGENTS COLLABORATIF               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│   │ AGENT MLP   │    │ AGENT CNN   │    │ AGENT RNN   │   │
│   │             │    │             │    │             │   │
│   │ Donnees     │    │ Images      │    │ Sequences   │   │
│   │ Tabulaires  │    │ 32x32x3     │    │ Texte       │   │
│   │ 13 features │    │ CIFAR-10    │    │ 20 tokens   │   │
│   │             │    │             │    │             │   │
│   │ Proba:      │    │ Proba:      │    │ Proba:      │   │
│   │ [0.7, 0.2,  │    │ [0.5, 0.3,  │    │ [0.6, 0.3,  │   │
│   │  0.1]       │    │  0.2]       │    │  0.1]       │   │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘   │
│          │                  │                   │          │
│          └──────────────────┼───────────────────┘          │
│                             ▼                              │
│                    ┌─────────────────┐                     │
│                    │  FUSION PIPELINE│                     │
│                    │                 │                     │
│                    │  Type: average  │                     │
│                    │                 │                     │
│                    │  [0.6, 0.27,    │                     │
│                    │   0.13]         │                     │
│                    └────────┬────────┘                     │
│                             ▼                              │
│                    ┌─────────────────┐                     │
│                    │  DECISION FINALE│                     │
│                    │                 │                     │
│                    │  Classe 0 (60%) │                     │
│                    │  Confiance: 95% │                     │
│                    └─────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""

print(architecture)

# Sauvegarder
torch.save({
    'agent_mlp': agent_mlp.state_dict(),
    'agent_cnn': agent_cnn.state_dict(),
    'agent_rnn': agent_rnn.state_dict(),
    'pipeline': pipeline.state_dict()
}, '../models/agents_collaboratifs.pth')

print("\nSysteme multi-agents sauvegarde!")