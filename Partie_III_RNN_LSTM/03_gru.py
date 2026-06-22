"""
==========================================
PARTIE III - ETAPE 3: GRU (Gated Recurrent Unit)
==========================================

GRU = Version simplifiee du LSTM
- Update gate: combine forget + input gates
- Reset gate: controle quoi oublier
- Pas de cell state separe

Avantages: moins de parametres, plus rapide que LSTM
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. DONNEES
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
# 2. ARCHITECTURE GRU
# ============================================================

print("=" * 60)
print("GRU - GATED RECURRENT UNIT")
print("=" * 60)

class CharGRU(nn.Module):
    """
    GRU pour prediction de caractere
    
    Differences avec LSTM:
    - 2 gates au lieu de 3 (update + reset)
    - Pas de cell state separe
    - Plus rapide et moins de parametres
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers=2, dropout=0.2):
        super(CharGRU, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.gru = nn.GRU(
            input_size, hidden_size, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
        
        out, hidden = self.gru(x, hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        
        return out, hidden


model = CharGRU(vocab_size, hidden_size=128, output_size=vocab_size, n_layers=2)
print(model)

total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parametres: {total:,}")


# ============================================================
# 3. ENTRAINEMENT
# ============================================================

print("\n" + "=" * 60)
print("ENTRAINEMENT GRU")
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
plt.title('Courbes de Loss - GRU')
plt.savefig('../outputs/gru_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 4. COMPARAISON RNN / LSTM / GRU
# ============================================================

print("\n" + "=" * 60)
print("COMPARAISON RNN / LSTM / GRU")
print("=" * 60)

comparison = """
| Modele | Parametres | Memoire | Vitesse | Complexite |
|--------|-----------|---------|---------|------------|
| RNN    | Faible    | Courte  | Rapide  | Simple     |
| LSTM   | Eleve     | Longue  | Lent    | Complexe   |
| GRU    | Moyen     | Longue  | Moyen   | Moyen      |

Conclusion:
- RNN: Sequences courtes, ressources limitees
- LSTM: Sequences longues, meilleure precision
- GRU: Bon compromis entre LSTM et RNN
"""

print(comparison)

torch.save(model, '../models/gru_complet.pth')
print("\nModele GRU sauvegarde: ../models/gru_complet.pth")