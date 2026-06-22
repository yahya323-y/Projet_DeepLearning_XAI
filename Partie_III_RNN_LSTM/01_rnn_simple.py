"""
==========================================
PARTIE III - ETAPE 1: RNN Simple
==========================================

Dataset: Predicteur de caractere (next character prediction)
Corpus: Texte simple pour demonstration

Concepts:
- RNN: Reseau recurrent qui traite les sequences
- BPTT: Backpropagation Through Time
- Perplexite: mesure de qualite du modele de langage
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. PREPARATION DES DONNEES TEXTUELLES
# ============================================================

print("=" * 60)
print("RNN SIMPLE - PREDICTION DE CARACTERE")
print("=" * 60)

# Corpus simple pour demonstration
text = """
Le deep learning est une branche de l'intelligence artificielle qui utilise des reseaux de neurones artificiels pour modeliser et resoudre des problemes complexes. Les reseaux de neurones profonds sont capables d'apprendre des representations hierarchiques des donnees.
"""

# Creer vocabulaire
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

print(f"Taille du vocabulaire: {vocab_size}")
print(f"Caracteres: {chars}")


# ============================================================
# 2. CREATION DES SEQUENCES
# ============================================================

seq_length = 20  # Longueur de la sequence d'entree

def create_sequences(text, seq_length):
    """Cree des sequences (input, target)"""
    sequences = []
    targets = []
    for i in range(len(text) - seq_length):
        seq = text[i:i + seq_length]
        target = text[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return sequences, targets

sequences, targets = create_sequences(text, seq_length)
print(f"Nombre de sequences: {len(sequences)}")


# ============================================================
# 3. ENCODAGE
# ============================================================

def encode_sequence(seq, char_to_idx):
    """Convertit une sequence en indices"""
    return [char_to_idx[ch] for ch in seq]

# Encoder toutes les sequences
X = torch.zeros((len(sequences), seq_length, vocab_size), dtype=torch.float32)
y = torch.zeros(len(sequences), dtype=torch.long)

for i, (seq, target) in enumerate(zip(sequences, targets)):
    encoded = encode_sequence(seq, char_to_idx)
    for t, idx in enumerate(encoded):
        X[i, t, idx] = 1.0  # One-hot encoding
    y[i] = char_to_idx[target]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================
# 4. DEFINITION DU RNN
# ============================================================

print("\n" + "=" * 60)
print("ARCHITECTURE RNN")
print("=" * 60)

class CharRNN(nn.Module):
    """
    RNN simple pour prediction de caractere
    
    Input: sequence de caracteres (one-hot)
    Hidden: etat cache qui memorise le contexte
    Output: caractere suivant predit
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Couche RNN
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        
        # Couche de sortie
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_size)
        # hidden: (n_layers, batch, hidden_size)
        
        if hidden is None:
            hidden = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
        
        # RNN forward
        out, hidden = self.rnn(x, hidden)
        # out: (batch, seq_len, hidden_size)
        
        # Prendre seulement le dernier pas de temps
        out = out[:, -1, :]  # (batch, hidden_size)
        
        # Couche de sortie
        out = self.fc(out)  # (batch, output_size)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)


model = CharRNN(vocab_size, hidden_size=128, output_size=vocab_size, n_layers=1)
print(model)

total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parametres: {total:,}")


# ============================================================
# 5. ENTRAINEMENT
# ============================================================

print("\n" + "=" * 60)
print("ENTRAINEMENT")
print("=" * 60)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Split train/val
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

N_EPOCHS = 500
history = {'train_loss': [], 'val_loss': []}

for epoch in range(N_EPOCHS):
    # Train
    model.train()
    optimizer.zero_grad()
    outputs, _ = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    
    # Gradient clipping (evite l'explosion des gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
    
    optimizer.step()
    
    # Val
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
plt.title('Courbes de Loss - RNN Simple')
plt.savefig('../outputs/rnn_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 6. GENERATION DE TEXTE
# ============================================================

print("\n" + "=" * 60)
print("GENERATION DE TEXTE")
print("=" * 60)

def generate_text(model, start_str, char_to_idx, idx_to_char, length=100):
    """Genere du texte a partir d'une sequence initiale"""
    model.eval()
    
    # Encoder la sequence initiale
    input_seq = torch.zeros(1, seq_length, vocab_size)
    encoded = encode_sequence(start_str[-seq_length:], char_to_idx)
    for t, idx in enumerate(encoded):
        input_seq[0, t, idx] = 1.0
    
    generated = start_str
    
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq)
            probs = torch.softmax(output, dim=1)
            
            # Echantillonner le caractere suivant
            next_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_idx]
            
            generated += next_char
            
            # Mettre a jour l'input
            input_seq = torch.zeros(1, seq_length, vocab_size)
            encoded = encode_sequence(generated[-seq_length:], char_to_idx)
            for t, idx in enumerate(encoded):
                input_seq[0, t, idx] = 1.0
    
    return generated

start = "Le deep learning"
generated = generate_text(model, start, char_to_idx, idx_to_char, length=100)
print(f"\nTexte genere:\n{generated}")

# Sauvegarder
torch.save(model, '../models/rnn_simple_complet.pth')
print("\nModele sauvegarde: ../models/rnn_simple_complet.pth")