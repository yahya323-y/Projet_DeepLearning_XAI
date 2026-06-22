"""
==========================================
PARTIE II - ETAPE 1: CNN Basique
==========================================

Dataset: CIFAR-10 (images 32x32x3, 10 classes)
Architecture: LeNet-like simplifie

Pourquoi CNN et pas MLP pour les images?
- MLP: 32*32*3 = 3072 features, trop de parametres
- CNN: utilise la localite spatiale, partage des poids, hierarchie des representations
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
# 1. CHARGEMENT ET PREPARATION DES DONNEES
# ============================================================

print("=" * 60)
print("1. CHARGEMENT CIFAR-10")
print("=" * 60)

# Transformations: normalisation des pixels [0,1] -> [-1,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Telechargement automatique
train_dataset = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform
)

# Split train -> train + val
train_size = int(0.85 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

# DataLoaders
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")

# Classes CIFAR-10
classes = ('avion', 'automobile', 'oiseau', 'chat', 'cerf',
           'chien', 'grenouille', 'cheval', 'bateau', 'camion')


# ============================================================
# 2. DEFINITION DU CNN
# ============================================================

class CNN_Basique(nn.Module):
    """
    CNN inspire de LeNet-5
    Architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> FC
    """
    
    def __init__(self):
        super(CNN_Basique, self).__init__()
        
        # Bloc convolutif 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # Bloc convolutif 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Bloc convolutif 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # Couches fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        # Bloc 1
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        # Bloc 2
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        # Bloc 3
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        # Aplatir
        x = x.view(x.size(0), -1)
        
        # FC
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================
# 3. ENTRAINEMENT
# ============================================================

def train_cnn(model, train_loader, val_loader, criterion, optimizer, n_epochs=50):
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\nEntrainement sur: {device}")
    
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
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Sauvegarder le meilleur modele
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '../models/best_cnn_basique.pth')
    
    print(f"\nMeilleure accuracy validation: {best_val_acc:.2f}%")
    return history


# ============================================================
# 4. EXECUTION
# ============================================================

if __name__ == "__main__":
    model = CNN_Basique()
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parametres: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = train_cnn(model, train_loader, val_loader, criterion, optimizer, n_epochs=30)
    
    # Courbes
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    
    plt.tight_layout()
    plt.savefig('../outputs/cnn_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Sauvegarder modele complet
    torch.save(model, '../models/cnn_basique_complet.pth')
    print("\nModele sauvegarde: ../models/cnn_basique_complet.pth")