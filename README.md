# 🧠 Deep Learning XAI - Projet EMSI

**Auteur:** Yahya Bark  
**École:** EMSI Casablanca  
**Année:** 2025-2026  
**Module:** Deep Learning

---

## 📋 Description

Ce projet présente un pipeline complet de Deep Learning avec une approche Explainable AI (XAI). Il couvre trois grandes familles d’architecture et une application de synthèse présentée dans un dashboard Streamlit.

- **Partie I : MLP + XAI** — classification tabulaire et interprétabilité SHAP
- **Partie II : CNN + Hybride CNN-LSTM** — traitement d’images et Grad-CAM
- **Partie III : RNN / LSTM / GRU** — modélisation de séquences et traduction Seq2Seq
- **Partie IV : Agents Collaboratifs** — fusion multi-modèles et méta-apprentissage
- **Partie V : Ablation Study** — évaluation de variantes et impact des composants
- **Partie VI : XAI Global** — dashboard global d’explicabilité

---

## 📁 Structure du projet

- `Partie_I_MLP_XAI/`
  - `01_preparation_donnees.py`
  - `02_mlp_sequential.py`
  - `03_mlp_custom.py`
  - `04_initialisation.py`
  - `05_xai_shap.py`
  - `06_ablation_mlp.py`

- `Partie_II_CNN_Hybride/`
  - `01_cnn_basique.py`
  - `02_cnn_lstm_hybride.py`
  - `03_gradcam_xai.py`
  - `04_ablation_cnn.py`

- `Partie_III_RNN_LSTM/`
  - `01_rnn_simple.py`
  - `02_lstm.py`
  - `03_gru.py`
  - `04_seq2seq_attention.py`
  - `05_beam_search.py`
  - `06_xai_attention.py`

- `Partie_IV_Agents_Collaboratifs/`
  - `01_pipeline_fusion.py`
  - `02_vote_ensemble.py`
  - `03_meta_learning.py`

- `Partie_V_Ablations/`
  - `ablation_study_complete.py`

- `Partie_VI_XAI_Global/`
  - `dashboard_xai.py`

- `Streamlit_App/`
  - `app.py`

- `outputs/` — résultats, graphiques et visualisations
- `models/` — modèles sauvegardés
- `data/` — données préparées

---

## 🚀 Installation et exécution

```bash
# Cloner le projet
git clone https://github.com/YAHYA_BARK/Projet_DeepLearning_XAI.git
cd Projet_DeepLearning_XAI
pip install -r requirements.txt
```

### Lancer l’application Streamlit

```bash
cd Streamlit_App
py -3.11 -m streamlit run app.py
```

Puis ouvrir le navigateur sur : `http://localhost:8501`

---

## ℹ️ Notes importantes

- `Streamlit_App/app.py` est une interface de présentation synthétique. Elle montre les 6 parties du projet de manière interactive, mais ne contient pas tout le code détaillé d’entraînement.
- Le code complet pour chaque partie se trouve dans les dossiers `Partie_I_MLP_XAI/`, `Partie_II_CNN_Hybride/`, `Partie_III_RNN_LSTM/`, `Partie_IV_Agents_Collaboratifs/`, `Partie_V_Ablations/` et `Partie_VI_XAI_Global/`.
- `README.md` et le dashboard servent à présenter rapidement l’architecture du projet au jury.

---

## 📌 Objectifs du projet

- Démontrer des architectures de Deep Learning sur plusieurs types de données
- Intégrer des techniques d’explicabilité XAI
- Comparer des variantes avec une étude d’ablation
- Montrer comment plusieurs modèles peuvent collaborer via une fusion d’agents
- Proposer un dashboard global pour visualiser les résultats

