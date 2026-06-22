"""
==========================================
PARTIE I - ETAPE 3: MLP avec Classe Personnalisee
==========================================

nn.Module = classe parent de PyTorch.
Permet de creer des modeles complexes avec controle total.

Ce qui est fait:
- Classe MLP_Custom avec initialisation Xavier
- Entrainement avec early stopping
- Evaluation avec metriques completes
- Sauvegarde du meilleur modele
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
# WineDataset class required to unpickle saved dataloaders
from torch.utils.data import Dataset


class WineDataset(Dataset):
    """
    Minimal WineDataset class to ensure pickle compatibility when
    loading `donnees_wine.pkl` which may have been pickled with
    this class defined in a different script's __main__ module.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 1. CLASSE MLP CUSTOM
# ============================================================

class MLP_Custom(nn.Module):
    """
    MLP personnalise avec:
    - Architecture configurable
    - Initialisation variable
    - Dropout pour regularisation
    """
    
    def __init__(self, n_features, n_classes, hidden_sizes=[64, 32], dropout=0.3, init_type='xavier'):
        """
        Args:
            n_features: nombre de features d'entree (13)
            n_classes: nombre de classes de sortie (3)
            hidden_sizes: tailles des couches cachees [64, 32]
            dropout: taux de dropout (0.3 = 30%)
            init_type: type d'initialisation ('xavier', 'gaussian', 'constant')
        """
        super(MLP_Custom, self).__init__()
        
        # Couche 1
        self.fc1 = nn.Linear(n_features, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Couche 2
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Couche de sortie
        self.fc3 = nn.Linear(hidden_sizes[1], n_classes)
        
        # Initialisation
        self.init_type = init_type
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation des poids selon le type choisi"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == 'gaussian':
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    nn.init.normal_(m.bias, mean=0, std=0.01)
                elif self.init_type == 'constant':
                    nn.init.constant_(m.weight, 0.1)
                    nn.init.constant_(m.bias, 0.1)
                elif self.init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Propagation avant (forward pass).
        Definit comment les donnees traversent le reseau.
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)  # Logits (pas de softmax ici - CrossEntropyLoss le fait)
        return x


# ============================================================
# 2. FONCTION D'ENTRAINEMENT
# ============================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                n_epochs=100, patience=10):
    """
    Entrainement avec early stopping.
    
    Early stopping = arreter si la validation ne s'ameliore plus
    pendant 'patience' epochs consecutives.
    """
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"\nEntrainement sur: {next(model.parameters()).device}")
    print("=" * 60)
    
    for epoch in range(n_epochs):
        # ===== MODE ENTRAINEMENT =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            # Forward
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistiques
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # ===== MODE VALIDATION =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Sauvegarder l'historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Afficher chaque 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Sauvegarder le meilleur modele
            torch.save(model.state_dict(), '../models/best_mlp_custom.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping a l'epoch {epoch+1}")
                break
    
    print(f"\nMeilleur modele sauvegarde: ../models/best_mlp_custom.pth")
    return history


# ============================================================
# 3. FONCTION D'EVALUATION
# ============================================================

def evaluate_model(model, test_loader):
    """Evaluation complete avec metriques"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.numpy())
            all_labels.extend(y_batch.numpy())
    
    # Metriques
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    print("\n" + "=" * 60)
    print("RESULTATS SUR TEST SET")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.ylabel('Vrai Label')
    plt.xlabel('Prediction')
    plt.savefig('../outputs/confusion_matrix_mlp.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return accuracy, precision, recall, f1


# ============================================================
# 4. EXECUTION PRINCIPALE
# ============================================================

if __name__ == "__main__":
    
    # Charger les donnees
    chemin_donnees = os.path.join('..', 'data', 'donnees_wine.pkl')
    with open(chemin_donnees, 'rb') as f:
        donnees = pickle.load(f)
    
    n_features = donnees['n_features']
    n_classes = donnees['n_classes']
    train_loader = donnees['train_loader']
    val_loader = donnees['val_loader']
    test_loader = donnees['test_loader']
    
    # Creer le modele
    model = MLP_Custom(n_features, n_classes, hidden_sizes=[64, 32], dropout=0.3)
    print(model)
    
    # Compter les parametres
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nNombre total de parametres: {total_params}")
    
    # Loss et Optimizer
    criterion = nn.CrossEntropyLoss()  # Pour classification multi-classe
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Entrainement
    history = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, 
        n_epochs=200, patience=20
    )
    
    # Courbes d'entrainement
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Courbes de Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Courbes d\'Accuracy')
    
    plt.tight_layout()
    plt.savefig('../outputs/training_curves_mlp.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Charger le meilleur modele et evaluer
    model.load_state_dict(torch.load('../models/best_mlp_custom.pth'))
    evaluate_model(model, test_loader)
    
    # Sauvegarder le modele complet
    torch.save(model, '../models/mlp_custom_complet.pth')
    print("\nModele complet sauvegarde: ../models/mlp_custom_complet.pth")
    