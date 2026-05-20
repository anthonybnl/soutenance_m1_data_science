# Projet Data Science M1 - Sujet 2 : Système Intelligent Multi-Modèles pour la Rétention Client et l'Évaluation du Risque de Revenus

## 📋 RÉSUMÉ DU BESOIN / CONTEXTE

**Contexte général :**
- Projet certifiant RNCP40875 (Bloc 2 - Expert en ingénierie de données)
- Construction d'un **système intelligent multi-modèles** pour la rétention client et l'évaluation du risque de revenus
- Environnement : entreprises SaaS, télécom, services par abonnement
- Dataset : 10 000 clients avec données comportementales (Kaggle customer_churn.csv)

**Objectif métier :**
Transformer des données clients brutes en système d'aide à la décision capable de :
- Anticiper les résiliations (churn)
- Évaluer l'impact financier
- Prioriser les actions de fidélisation

---

## 🎯 RÉSUMÉ DES ATTENDUS

### **Livrables obligatoires :**
1. **Code source fonctionnel** avec documentation technique
2. **Rapport de projet** structuré
3. **Support de présentation**
4. **Dashboard interactif** (Streamlit/Dash)
5. **Présentation + démonstration** en classe (évaluation individuelle)

### **Composantes techniques obligatoires :**
- ✅ **4 modèles minimum** (ML classiques + Deep Learning)
- ✅ **1 modèle Deep Learning obligatoire** (MLP)
- ✅ **1 tâche prédictive** (classification OU régression au choix)
- ✅ **Dashboard interactif décisionnel**
- ✅ **Comparaison rigoureuse** des modèles
- ✅ **Interprétabilité** (Feature Importance / SHAP)
- ⭕ **API REST** (optionnelle mais valorisante)

### **Tâches prédictives possibles :**
1. **Prédiction du Churn** (classification binaire) ← recommandée
2. Estimation du revenu à risque (régression)
3. Estimation CLV (Customer Lifetime Value)
4. Score d'engagement client

---

## ✅ LISTE EXPLICITE DES EXIGENCES & RECOMMANDATIONS

### **EXIGENCES FONCTIONNELLES (EF)**

#### **EF1 : Acquisition et Préparation des Données**
- Nettoyage des valeurs manquantes
- Encodage des variables catégorielles
- Normalisation/standardisation
- Analyse exploratoire documentée (EDA)

#### **EF2 : Modélisation Multi-Algorithmes**
- ✅ Minimum **4 modèles** (classification et/ou régression)
- ✅ **Au moins 1 modèle Deep Learning** (MLP obligatoire)
- ✅ Modèles de référence + modèles avancés
- ✅ Sélection et justification d'un **modèle candidat final**

#### **EF3 : Système d'Évaluation**
- Métriques adaptées :
  - **Classification** : Accuracy, Precision, Recall, F1, ROC-AUC
  - **Régression** : MAE, RMSE, R²
- Tableaux et graphes comparatifs
- Analyse des erreurs

#### **EF4 : Dashboard Interactif (OBLIGATOIRE)**
- Saisie de scénarios clients
- Affichage des prédictions
- Comparaison des modèles
- Importance des variables
- Graphiques interactifs
- Framework : Streamlit (recommandé) ou Dash/Plotly

#### **EF5 : API REST (OPTIONNELLE)**
- POST `/predict` : prédiction sur nouvelles données
- GET `/health` : vérification du service
- Gestion des erreurs
- Documentation (README ou Swagger/FastAPI)

---

### **RECOMMANDATIONS MÉTHODOLOGIQUES**

#### **R1 : Analyse Exploratoire Approfondie (EDA)**
- Comprendre les distributions
- Identifier les valeurs extrêmes, anomalies
- Analyser les relations entre variables
- Détecter les problèmes invisibles

#### **R2 : Vérification des Distributions de Classes**
- ⚠️ Déséquilibre des classes fréquent
- Privilégier métriques adaptées : Recall, F1, PR-AUC
- Stratégies : stratified split, class_weight, ajustement de seuil

#### **R3 : Analyse de Corrélation et Redondance**
- Étudier les corrélations entre features
- Identifier les variables redondantes
- Limiter la multicolinéarité
- Créer des variables dérivées pertinentes (ratio, évolutions)

#### **R4 : Approche Progressive - Baseline d'abord**
- ✅ **Étape 1** : Modèle baseline simple (régression logistique)
- ✅ **Étape 2** : Modèles plus complexes (RF, GB, SVM)
- ✅ **Étape 3** : Deep Learning (MLP)
- Justifier chaque gain de complexité

#### **R5 : Comparaison Systématique ≥ 4 Modèles**
- Implémenter minimum 4 modèles
- Comparer de manière structurée
- Sélectionner un **modèle candidat final**
- Argumenter le choix : performance + stabilité + interprétabilité + coût

#### **R6 : Éviter le Data Leakage (IMPÉRATIF)**
- ⚠️ Preprocessing uniquement sur train set
- Appliquer ensuite au test set
- Utiliser sklearn.Pipeline ou ColumnTransformer
- Erreur = invalidité totale des résultats

#### **R7 : Validation Croisée (Cross-Validation)**
- Utiliser CV pour robustesse
- Moins de dépendance au split unique
- Meilleure comparaison des modèles

#### **R8 : Optimisation des Hyperparamètres**
- GridSearch ou RandomizedSearch
- Expliquer la stratégie
- Plages réalistes
- Éviter le tuning excessif non justifié

#### **R9 : Analyse des Erreurs**
- Matrices de confusion (classification)
- Analyse des résidus (régression)
- Étudier les cas mal prédits
- Discuter les raisons possibles

#### **R10 : Interprétabilité (CRUCIAL)**
- **Feature Importance** :
  - `model.feature_importances_` (arbres)
  - Permutation Importance (recommandée)
- **SHAP** (avancé) :
  - Explicabilité locale (pourquoi ce client ?)
  - Explicabilité globale (variables importantes)
  - Impact positif/négatif de chaque variable

#### **R11 : Structuration du Code**
- ✅ Organiser en **modules** :
  - `data preprocessing`
  - `modeling`
  - `evaluation`
  - `API` (optionnel)
  - `dashboard`
- Éviter le notebook monolithique
- Séparer les responsabilités
- Structure professionnelle maintenable

#### **R12 : Versionnement Git**
- Utiliser Git régulièrement
- Commits explicites et fréquents
- Branches si travail en équipe
- Historique propre = maturité professionnelle

#### **R13 : Dashboard Indépendant et Orienté Décision**
- Outil décisionnel autonome
- Visualisation des données
- Exploration des indicateurs clés
- Comparaison des modèles
- Prédictions sur scénarios utilisateur
- Utilisable par profil métier non-technique

#### **R14 : API Testée Indépendamment (si optionnel choisi)**
- Valider fonctionnement avant intégration dashboard
- Tester avec Postman/curl/script Python
- Vérifier endpoints et gestion d'erreurs
- Dashboard = couche de présentation, pas compensation

#### **R15 : Posture Professionnelle**
- Documenter les choix
- Justifier les décisions
- Argumenter les comparaisons
- Démarche scientifique structurée
- Discuter les limites et risques
- Recommandations opérationnelles

---

## 📝 TODO LISTE INITIALE 

Created 15 todos

---

## 📁 PROPOSITION D'ARBORESCENCE PROJET

```
soutenance_m1_data_science/
│
├── README.md                          # Documentation principale du projet
├── requirements.txt                   # Dépendances Python
├── .gitignore                         # Fichiers à ignorer (Git)
│
├── doc/                               # Documentation
│   ├── Projet M1 DE Sujet 2.md       # ✅ Cahier des charges (déjà présent)
│   ├── rapport_projet.md             # Rapport final
│   └── presentation.pdf              # Support de présentation
│
├── data/                              # Données
│   ├── raw/                          # Données brutes
│   │   └── customer_churn.csv        # Dataset Kaggle
│   ├── processed/                    # Données transformées
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── external/                     # Données externes (optionnel)
│
├── notebooks/                         # Notebooks d'exploration
│   ├── 01_EDA.ipynb                  # Analyse exploratoire
│   ├── 02_feature_engineering.ipynb  # Création de features
│   ├── 03_modeling_baseline.ipynb    # Modèle baseline
│   ├── 04_modeling_advanced.ipynb    # Modèles avancés
│   ├── 05_deep_learning.ipynb        # Modèle MLP
│   └── 06_interpretability.ipynb     # SHAP & Feature Importance
│
├── src/                               # Code source modulaire
│   ├── __init__.py
│   │
│   ├── data/                         # Module préparation données
│   │   ├── __init__.py
│   │   ├── load_data.py              # Chargement données
│   │   ├── preprocessing.py          # Nettoyage & transformation
│   │   └── feature_engineering.py    # Création de features
│   │
│   ├── models/                       # Module modélisation
│   │   ├── __init__.py
│   │   ├── baseline.py               # Régression logistique
│   │   ├── ensemble.py               # Random Forest, XGBoost
│   │   ├── deep_learning.py          # MLP (PyTorch/Keras)
│   │   └── model_trainer.py          # Pipeline d'entraînement
│   │
│   ├── evaluation/                   # Module évaluation
│   │   ├── __init__.py
│   │   ├── metrics.py                # Calcul des métriques
│   │   ├── visualization.py          # Graphiques comparatifs
│   │   └── interpretability.py       # SHAP & Feature Importance
│   │
│   ├── api/                          # Module API (optionnel)
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app
│   │   ├── schemas.py                # Modèles Pydantic
│   │   └── model_service.py          # Service prédiction
│   │
│   └── dashboard/                    # Module Dashboard
│       ├── __init__.py
│       ├── app.py                    # Application Streamlit
│       ├── components.py             # Composants UI
│       └── utils.py                  # Fonctions utilitaires
│
├── models/                            # Modèles entraînés sauvegardés
│   ├── baseline_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── mlp_model.h5
│   └── best_model.pkl                # Modèle final sélectionné
│
├── outputs/                           # Résultats & visualisations
│   ├── figures/                      # Graphiques générés
│   │   ├── eda/
│   │   ├── model_comparison/
│   │   └── interpretability/
│   ├── reports/                      # Rapports automatiques
│   │   └── model_comparison.csv
│   └── logs/                         # Logs d'exécution
│
├── tests/                             # Tests unitaires (optionnel)
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_api.py
│
├── scripts/                           # Scripts d'exécution
│   ├── download_data.sh              # Téléchargement dataset
│   ├── train_all_models.py           # Entraînement complet
│   ├── run_dashboard.sh              # Lancement dashboard
│   └── run_api.sh                    # Lancement API (optionnel)
│
└── config/                            # Configurations
    ├── config.yaml                   # Config générale
    ├── model_params.yaml             # Hyperparamètres
    └── paths.yaml                    # Chemins des fichiers
```

---

## 🎯 RÉSUMÉ DES POINTS CRITIQUES

### ⚠️ **Pièges à éviter absolument :**
1. **Data Leakage** → Invalidité totale du projet
2. **Déséquilibre des classes** → Métriques inadaptées
3. **Pas de comparaison multi-modèles** → Non-respect du cahier des charges
4. **Absence de Deep Learning** → Exigence non remplie
5. **Dashboard non fonctionnel** → Livrable obligatoire manquant

### ✅ **Critères de réussite :**
- ≥ 4 modèles comparés rigoureusement
- 1 modèle Deep Learning (MLP)
- Dashboard interactif exploitable
- Interprétabilité des modèles (SHAP/Feature Importance)
- Code structuré et versionné
- Rapport professionnel argumenté
- Présentation convaincante avec démonstration

---

**Le projet est maintenant clairement cartographié ! Prêt à démarrer ? 🚀**