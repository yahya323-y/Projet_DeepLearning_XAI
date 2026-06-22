# 🧠 Deep Learning XAI - Projet EMSI

**Auteur:** Yahya Bark  
**École:** EMSI Casablanca  
**Année:** 2025-2026  
**Module:** Deep Learning

---

## 📋 Description

Projet de fin de module Deep Learning couvrant trois architectures fondamentales avec Explainable AI (XAI):

- **MLP** (Perceptron Multi-Couche) pour données tabulaires
- **CNN** (Réseaux Convolutifs) + Hybride CNN-LSTM pour images
- **RNN/LSTM/GRU** + Seq2Seq pour séquences textuelles

Innovations:
- 🤖 **Agents Collaboratifs** (fusion multi-modèles)
- 🔥 **Ablation Study** (étude d'impact des composantes)
- 🔍 **XAI Global** (SHAP, Grad-CAM, Attention)

---

## 📁 Structure du Projet
Projet_DeepLearning_XAI/
├── Partie_I_MLP_XAI/              # MLP + XAI (SHAP)
│   ├── 01_preparation_donnees.py
│   ├── 02_mlp_sequential.py
│   ├── 03_mlp_custom.py
│   ├── 04_initialisation.py
│   ├── 05_xai_shap.py
│   └── 06_ablation_mlp.py
│
├── Partie_II_CNN_Hybride/          # CNN + CNN-LSTM
│   ├── 01_cnn_basique.py
│   ├── 02_cnn_lstm_hybride.py
│   ├── 03_gradcam_xai.py
│   └── 04_ablation_cnn.py
│
├── Partie_III_RNN_LSTM/          # RNN/LSTM/GRU + Seq2Seq
│   ├── 01_rnn_simple.py
│   ├── 02_lstm.py
│   ├── 03_gru.py
│   ├── 04_seq2seq_attention.py
│   ├── 05_beam_search.py
│   └── 06_xai_attention.py
│
├── Partie_IV_Agents_Collaboratifs/ # Fusion multi-agents
│   ├── 01_pipeline_fusion.py
│   ├── 02_vote_ensemble.py
│   └── 03_meta_learning.py
│
├── Partie_V_Ablations/             # Ablation study global
│   └── ablation_study_complete.py
│
├── Partie_VI_XAI_Global/          # Dashboard XAI
│   └── dashboard_xai.py
│
├── Streamlit_App/                  # Application interactive
│   └── app_complet.py
│
├── outputs/                        # Résultats et visualisations
├── models/                         # Modèles sauvegardés
└── data/                           # Données préparées
