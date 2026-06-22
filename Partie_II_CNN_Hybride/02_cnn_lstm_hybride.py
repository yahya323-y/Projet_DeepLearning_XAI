"""
==========================================
PARTIE II - ETAPE 2: Modele Hybride CNN-LSTM
==========================================

Innovation: Combiner CNN (extraction spatiale) + LSTM (modelisation sequentielle)

Idee: Traiter l'image comme une sequence de regions
- CNN extrait les features de chaque region
- LSTM modelise les dependances entre regions

Architecture:
Image (32x32x3) -> CNN Encoder -> Sequence de features -> LSTM -> Classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os


# ============================================================
# 1. CHARGEMENT DES DONNEES
# ============================================================

print("=" * 60)
print("CNN-LSTM HYBRIDE - CIFAR-10")
print("=" * 60)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform
)

train_size = int(0.85 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('avion', 'automobile', 'oiseau', 'chat', 'cerf',
           'chien', 'grenouille', 'cheval', 'bateau', 'camion')


# ============================================================
# 2. MODELE HYBRIDE CNN-LSTM
# ============================================================

class CNN_LSTM_Hybride(nn.Module):
    """
    CNN-LSTM Hybride pour classification d'images.
    
    Idee: Diviser l'image en patches, extraire features avec CNN,
    puis modeliser la sequence avec LSTM.
    """
    
    def __init__(self, n_classes=10, hidden_lstm=128, num_layers=2):
        super(CNN_LSTM_Hybride, self).__init__()
        
        # === CNN ENCODER ===
        # Extrait les features spatiales
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32->16
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16->8
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8->4
        
        # === TRANSFORMATION EN SEQUENCE ===
        # 128 canaux * 4x4 = 128 features par position spatiale
        # On traite chaque position (4x4=16 positions) comme un timestep
        self.feature_dim = 128  # Dimension des features CNN
        
        # === LSTM ===
        # Modelise les dependances spatiales comme une sequence
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_lstm,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # === CLASSIFICATION ===
        self.fc = nn.Linear(hidden_lstm, n_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # === CNN ENCODER ===
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))  # (B, 32, 16, 16)
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))  # (B, 64, 8, 8)
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))   # (B, 128, 4, 4)
        
        # === TRANSFORMER EN SEQUENCE ===
        # (B, 128, 4, 4) -> (B, 128, 16) -> (B, 16, 128)
        # 16 positions spatiales, 128 features chacune
        x = x.view(batch_size, self.feature_dim, -1)  # (B, 128, 16)
        x = x.permute(0, 2, 1)  # (B, 16, 128) -> (batch, seq_len, features)
        
        # === LSTM ===
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Prendre le dernier hidden state
        last_hidden = hidden[-1]  # (B, hidden_lstm)
        
        # === CLASSIFICATION ===
        out = self.fc(last_hidden)
        return out


# ============================================================
# 3. ENTRAINEMENT
# ============================================================

def train_hybrid(model, train_loader, val_loader, criterion, optimizer, n_epochs=30):
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\nEntrainement sur: {device}")
    print("=" * 60)
    
    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '../models/best_cnn_lstm_hybride.pth')
    
    return history


# ============================================================
# 4. EXECUTION ET COMPARAISON
# ============================================================

if __name__ == "__main__":
    model = CNN_LSTM_Hybride(n_classes=10, hidden_lstm=128, num_layers=2)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parametres CNN-LSTM: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = train_hybrid(model, train_loader, val_loader, criterion, optimizer, n_epochs=30)
    
    # Courbes
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title('CNN-LSTM: Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.title('CNN-LSTM: Accuracy')
    
    plt.tight_layout()
    plt.savefig('../outputs/cnn_lstm_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Sauvegarder
    torch.save(model, '../models/cnn_lstm_hybride_complet.pth')
    print("\nModele hybride sauvegarde!")
    
    print("\n" + "=" * 60)
    print("COMPARAISON CNN vs CNN-LSTM")
    print("=" * 60)
    print("CNN Basique: Extraction spatiale pure")
    print("CNN-LSTM: Extraction spatiale + Modelisation sequentielle")
    print("Le hybride capture mieux les relations spatiales longue distance")