"""
==========================================
PARTIE II - ETAPE 3: Grad-CAM XAI
==========================================

Grad-CAM = Gradient-weighted Class Activation Mapping
Visualise les zones de l'image qui ont influence la decision.

Rouge = zone tres importante
Bleu = zone peu importante
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


# ============================================================
# 1. CHARGER MODELE ET DONNEES
# ============================================================

print("=" * 60)
print("GRAD-CAM: VISUALISATION DES ZONES IMPORTANTES")
print("=" * 60)

# Charger le modele CNN
chemin_modele = os.path.join('..', 'models', 'cnn_basique_complet.pth')
model = torch.load(chemin_modele)
model.eval()

# Donnees
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

classes = ('avion', 'automobile', 'oiseau', 'chat', 'cerf',
           'chien', 'grenouille', 'cheval', 'bateau', 'camion')


# ============================================================
# 2. CLASSE GRAD-CAM
# ============================================================

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    
    Principe:
    1. Forward pass jusqu'a la derniere couche convolutive
    2. Calculer les gradients de la classe cible par rapport aux feature maps
    3. Moyenne des gradients = poids d'importance
    4. Combinaison ponderee des feature maps = heatmap
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks pour capturer gradients et activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        # Forward
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pour la classe cible
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Poids = moyenne des gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # CAM = combinaison ponderee
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU
        
        # Normalisation
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy(), target_class


# ============================================================
# 3. VISUALISATION
# ============================================================

def visualize_gradcam(model, test_loader, n_images=5):
    """
    Visualise Grad-CAM sur plusieurs images
    """
    # Trouver la derniere couche convolutive
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module
    
    gradcam = GradCAM(model, target_layer)
    
    fig, axes = plt.subplots(n_images, 3, figsize=(12, 4*n_images))
    
    for idx in range(n_images):
        # Recuperer une image
        images, labels = next(iter(test_loader))
        image = images[0]
        
        # Generer CAM
        cam, pred_class = gradcam.generate_cam(images)
        
        # Redimensionner CAM a la taille de l'image
        cam = cv2.resize(cam, (32, 32))
        
        # Convertir image pour affichage
        img_display = image.permute(1, 2, 0).numpy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        
        # Superposition
        heatmap = plt.cm.jet(cam)[:, :, :3]
        superimposed = img_display * 0.6 + heatmap * 0.4
        
        # Affichage
        axes[idx, 0].imshow(img_display)
        axes[idx, 0].set_title(f'Image originale\nVrai: {classes[labels[0]]}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(cam, cmap='jet')
        axes[idx, 1].set_title('Grad-CAM Heatmap')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(superimposed)
        axes[idx, 2].set_title(f'Superposition\nPred: {classes[pred_class]}')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('../outputs/gradcam_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nVisualisation Grad-CAM sauvegardee!")


# ============================================================
# 4. EXECUTION
# ============================================================

if __name__ == "__main__":
    visualize_gradcam(model, test_loader, n_images=5)
    
    print("\nInterpretation Grad-CAM:")
    print("- Rouge/Jaune: Zones tres importantes pour la classification")
    print("- Bleu: Zones peu importantes")
    print("- Permet de verifier que le modele regarde les bonnes zones")
    print("- Si le modele regarde le fond plutot que l'objet = probleme!")