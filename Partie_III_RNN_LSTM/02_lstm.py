"""
==========================================
PARTIE III - ETAPE 2: LSTM (Long Short-Term Memory)
==========================================

LSTM = RNN ameliore avec mecanisme de memoire
- Cell state: memoire a long terme
- Forget gate: oublier l'information inutile
- Input gate: stocker nouvelle information
- Output gate: decider quoi sortir

Compare au RNN simple: meilleure memoire, moins de vanishing gradient
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. DONNEES (meme corpus)
# ============================================================

text = """
Le deep learning est une branche de l'intelligence artificielle qui utilise des reseaux de neurones artificiels pour modeliser et resoudre des problemes complexes. Les reseaux de neurones profonds sont capables d'apprendre des representations hierarchiques des donnees.
"""

chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

seq_length = 20

def create_sequences(text, seq_length):
    sequences, targets = [], []
    for i in range(len(text) - seq_length):
        sequences.append(text[i:i + seq_length])
        targets.append(text[i + seq_length])
    return sequences, targets

sequences, targets = create_sequences(text, seq_length)

def encode_sequence(seq, char_to_idx):
    return [char_to_idx[ch] for ch in seq]

X = torch.zeros((len(sequences), seq_length, vocab_size), dtype=torch.float32)
y = torch.zeros(len(sequences), dtype=torch.long)

for i, (seq, target) in enumerate(zip(sequences, targets)):
    encoded = encode_sequence(seq, char_to_idx)
    for t, idx in enumerate(encoded):
        X[i, t, idx] = 1.0
    y[i] = char_to_idx[target]


# ============================================================
# 2. ARCHITECTURE LSTM
# ============================================================

print("=" * 60)
print("LSTM - LONG SHORT-TERM MEMORY")
print("=" * 60)

class CharLSTM(nn.Module):
    """
    LSTM pour prediction de caractere
    
    Avantages vs RNN:
    - Cell state: memoire a long terme stable
    - Gates: controle du flux d'information
    - Moins de vanishing gradient
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers=2, dropout=0.2):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Couche LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        
        # Couche de sortie
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        if hidden is None:
            h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
            hidden = (h0, c0)
        
        # LSTM forward
        out, hidden = self.lstm(x, hidden)
        
        # Dernier pas de temps
        out = out[:, -1, :]
        
        # Sortie
        out = self.fc(out)
        
        return out, hidden


model = CharLSTM(vocab_size, hidden_size=128, output_size=vocab_size, n_layers=2)
print(model)

total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parametres: {total:,}")


# ============================================================
# 3. ENTRAINEMENT
# ============================================================

print("\n" + "=" * 60)
print("ENTRAINEMENT LSTM")
print("=" * 60)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

N_EPOCHS = 500
history = {'train_loss': [], 'val_loss': []}

for epoch in range(N_EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs, _ = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_outputs, _ = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    
    history['train_loss'].append(loss.item())
    history['val_loss'].append(val_loss.item())
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{N_EPOCHS}] "
              f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

# Courbes
plt.figure(figsize=(10, 4))
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Courbes de Loss - LSTM')
plt.savefig('../outputs/lstm_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 4. COMPARAISON RNN vs LSTM
# ============================================================

print("\n" + "=" * 60)
print("COMPARAISON RNN vs LSTM")
print("=" * 60)

print("RNN Simple:")
print("  + Simple, moins de parametres")
print("  - Vanishing gradient, memoire courte")
print("  - Difficile d'apprendre les dependances longues")

print("\nLSTM:")
print("  + Memoire longue grace au cell state")
print("  + Gates controlent l'information")
print("  - Plus complexe, plus de parametres")
print("  - Meilleure pour les sequences longues")

# Sauvegarder
torch.save(model, '../models/lstm_complet.pth')
print("\nModele LSTM sauvegarde: ../models/lstm_complet.pth")