import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os


# ============================================================
# 1. CHARGEMENT DU DATASET
# ============================================================

print("=" * 60)
print("1. CHARGEMENT DU DATASET WINE")
print("=" * 60)

data = load_wine()
X = data.data      # Features: (178, 13)
y = data.target    # Labels: (178,) - classes 0, 1, 2

print(f"Nombre d'echantillons: {X.shape[0]}")
print(f"Nombre de features: {X.shape[1]}")
print(f"Classes: {np.unique(y)}")
print(f"Noms des features: {data.feature_names}")


# ============================================================
# 2. SPLIT DES DONNEES
# ============================================================

print("\n" + "=" * 60)
print("2. SPLIT: TRAIN / VALIDATION / TEST")
print("=" * 60)

# stratify=y = garantir que chaque classe est presente dans chaque split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
print(f"Validation: {len(X_val)} ({len(X_val)/len(X)*100:.0f}%)")
print(f"Test: {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")


# ============================================================
# 3. NORMALISATION
# ============================================================

print("\n" + "=" * 60)
print("3. NORMALISATION (StandardScaler)")
print("=" * 60)

# StandardScaler: moyenne = 0, ecart-type = 1
# IMPORTANT: fit uniquement sur le train set pour eviter le data leakage

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Avant normalisation (train):")
print(f"  Moyenne: {X_train.mean(axis=0)[:3]}...")
print(f"  Ecart-type: {X_train.std(axis=0)[:3]}...")

print("\nApres normalisation (train):")
print(f"  Moyenne: {X_train_scaled.mean(axis=0)[:3]}... (proche de 0)")
print(f"  Ecart-type: {X_train_scaled.std(axis=0)[:3]}... (proche de 1)")


# ============================================================
# 4. CONVERSION EN TENSORS PYTORCH
# ============================================================

print("\n" + "=" * 60)
print("4. CONVERSION EN TENSORS PYTORCH")
print("=" * 60)

# FloatTensor = nombres decimaux (features)
# LongTensor = entiers (labels/classes)
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)

X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.LongTensor(y_val)

X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

print(f"X_train_tensor: {X_train_tensor.shape} (type: {X_train_tensor.dtype})")
print(f"y_train_tensor: {y_train_tensor.shape} (type: {y_train_tensor.dtype})")


# ============================================================
# 5. DATASET PERSONNALISE PYTORCH
# ============================================================

class WineDataset(Dataset):
    """
    Dataset personnalise pour PyTorch.
    
    3 methodes obligatoires:
    - __init__: initialisation
    - __len__: nombre d'echantillons
    - __getitem__: recuperer un echantillon par index
    """
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Creation des datasets
train_dataset = WineDataset(X_train_tensor, y_train_tensor)
val_dataset = WineDataset(X_val_tensor, y_val_tensor)
test_dataset = WineDataset(X_test_tensor, y_test_tensor)

print(f"\nDataset train: {len(train_dataset)} echantillons")
print(f"Dataset val: {len(val_dataset)} echantillons")
print(f"Dataset test: {len(test_dataset)} echantillons")


# ============================================================
# 6. DATALOADER (MINI-BATCHES)
# ============================================================

print("\n" + "=" * 60)
print("6. DATALOADER (MINI-BATCHES)")
print("=" * 60)

BATCH_SIZE = 32

# shuffle=True pour le train = melanger les donnees a chaque epoch
# shuffle=False pour val/test = evaluation stable
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Taille du batch: {BATCH_SIZE}")
print(f"Nombre de batches (train): {len(train_loader)}")
print(f"Nombre de batches (val): {len(val_loader)}")
print(f"Nombre de batches (test): {len(test_loader)}")


# ============================================================
# 7. SAUVEGARDE
# ============================================================

print("\n" + "=" * 60)
print("7. SAUVEGARDE DES DONNEES")
print("=" * 60)

donnees = {
    'X_train': X_train_tensor,
    'y_train': y_train_tensor,
    'X_val': X_val_tensor,
    'y_val': y_val_tensor,
    'X_test': X_test_tensor,
    'y_test': y_test_tensor,
    'train_loader': train_loader,
    'val_loader': val_loader,
    'test_loader': test_loader,
    'scaler': scaler,
    'feature_names': data.feature_names,
    'n_features': X.shape[1],
    'n_classes': len(np.unique(y))
}

# Sauvegarder dans le dossier parent (accessible par tous les scripts)
chemin_sauvegarde = os.path.join('..', 'data', 'donnees_wine.pkl')

with open(chemin_sauvegarde, 'wb') as f:
    pickle.dump(donnees, f)

print(f"✅ Donnees sauvegardees: {chemin_sauvegarde}")
print(f"\nResume:")
print(f"  Features: {X.shape[1]}")
print(f"  Classes: {len(np.unique(y))}")
print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print("\nPret pour l'entrainement!")