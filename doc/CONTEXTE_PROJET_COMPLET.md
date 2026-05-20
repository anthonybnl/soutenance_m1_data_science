# 📋 CONTEXTE COMPLET DU PROJET - Système Intelligent Multi-Modèles pour la Rétention Client

> **Date de création :** 20 mai 2026  
> **Projet :** M1 Data Science - Certification RNCP40875 (Bloc 2)  
> **Étudiant :** Anto

---

## 🎯 RÉSUMÉ EXÉCUTIF

### Objectif principal
Développer un **système intelligent multi-modèles** capable de prédire le churn client (résiliation d'abonnement) et d'évaluer le risque financier associé pour des entreprises SaaS, télécom ou services par abonnement.

### Dataset
- **Source :** [Kaggle - Customer Churn Business Dataset](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset)
- **Taille :** 10 000 clients (observations)
- **Variables :** 31 colonnes (30 features + 1 target)
- **Types :** Variables numériques et catégorielles
- **Variable cible :** `churn` (0 = Fidèle, 1 = Parti)

### Tâche prédictive choisie
**Classification binaire : Prédiction du Churn**  
Identifier les clients susceptibles de résilier leur abonnement pour permettre des actions de rétention ciblées.

---

## 📊 STRUCTURE DU DATASET

### Variables disponibles

#### **Variables démographiques :**
- `customer_id` : Identifiant unique
- `gender` : Genre (Male, Female)
- `country` : Pays du client
- `city` : Ville du client
- `customer_segment` : Segment de clientèle

#### **Variables d'engagement produit :**
- `tenure_months` : Ancienneté du client (mois)
- `monthly_logins` : Nombre de connexions mensuelles
- `weekly_active_days` : Nombre de jours actifs par semaine
- `avg_session_time` : Temps moyen de session
- `features_used` : Nombre de fonctionnalités utilisées
- `usage_growth_rate` : Taux de croissance de l'utilisation
- `last_login_days_ago` : Nombre de jours depuis la dernière connexion

#### **Variables contractuelles :**
- `signup_channel` : Canal d'inscription (Web, Mobile, etc.)
- `contract_type` : Type de contrat (Monthly, Annual, Two-Year)

#### **Variables financières :**
- `monthly_fee` : Frais mensuels ($)
- `total_revenue` : Revenu total généré ($)
- `payment_method` : Méthode de paiement
- `payment_failures` : Nombre d'échecs de paiement
- `discount_applied` : Montant total des réductions ($)
- `price_increase_last_3m` : Augmentation de prix récente (true/false)

#### **Variables support client :**
- `support_tickets` : Nombre de tickets ouverts
- `avg_resolution_time` : Temps moyen de résolution (heures)
- `complaint_type` : Type de plainte
- `escalations` : Nombre d'escalades

#### **Variables satisfaction & marketing :**
- `csat_score` : Score de satisfaction client
- `nps_score` : Net Promoter Score (0-100)
- `survey_response` : Taux de réponse aux enquêtes (%)
- `email_open_rate` : Taux d'ouverture emails (%)
- `marketing_click_rate` : Taux de clics campagnes (%)
- `referral_count` : Nombre de parrainages

#### **Variable cible :**
- `churn` : 0 = Client fidèle, 1 = Client parti

---

## ✅ LIVRABLES OBLIGATOIRES

### 1. Code source fonctionnel
- Structure modulaire (pas de notebook monolithique)
- Pipeline de preprocessing sécurisé (éviter data leakage)
- Documentation technique

### 2. Rapport de projet
- Démarche méthodologique structurée
- Justification des choix techniques
- Analyse critique des résultats
- Discussion des limites
- Recommandations opérationnelles

### 3. Support de présentation
- Présentation professionnelle du projet
- Démonstration en classe
- Évaluation individuelle de tous les membres

### 4. Dashboard interactif ⭐ (OBLIGATOIRE)
- Framework : **Streamlit** (recommandé) ou Dash/Plotly
- Fonctionnalités requises :
  - Saisie de scénarios clients
  - Affichage des prédictions
  - Comparaison des modèles
  - Importance des variables
  - Graphiques interactifs
  - Visuels KPI métier (clients à risque, revenu à risque)

### 5. API REST (OPTIONNELLE mais valorisante)
- Framework : FastAPI ou Flask
- Endpoints requis :
  - `POST /predict` : Prédiction sur nouvelles données
  - `GET /health` : Vérification du service
- Gestion des erreurs
- Documentation (README ou Swagger)

---

## 🔧 EXIGENCES TECHNIQUES OBLIGATOIRES

### EF1 : Préparation des données
- ✅ Nettoyage des valeurs manquantes
- ✅ Encodage des variables catégorielles
- ✅ Normalisation/standardisation
- ✅ Analyse exploratoire documentée (EDA)

### EF2 : Modélisation Multi-Algorithmes ⭐ CRITIQUE
- ✅ **Minimum 4 modèles** (classification)
- ✅ **Au moins 1 modèle Deep Learning** (MLP obligatoire)
- ✅ Modèles baseline + modèles avancés
- ✅ Sélection et justification d'un **modèle candidat final**

**Exemples de modèles à implémenter :**
1. **Baseline :** Régression Logistique
2. **Ensemble :** Random Forest
3. **Boosting :** XGBoost ou LightGBM ou Gradient Boosting
4. **Deep Learning :** MLP (Multi-Layer Perceptron) - OBLIGATOIRE
5. **Bonus :** SVM, autres modèles

### EF3 : Système d'Évaluation
**Métriques prioritaires (déséquilibre des classes) :**
- ✅ **Recall** (détecter les vrais churners)
- ✅ **F1-Score** (compromis precision/recall)
- ✅ **Precision** (éviter faux positifs)
- ✅ **ROC-AUC** ou **PR-AUC** (performance globale)
- ⚠️ **Accuracy** (métrique secondaire à cause du déséquilibre)

**Livrables évaluation :**
- Tableaux comparatifs des métriques
- Graphiques de comparaison (barplots, courbes ROC)
- Matrices de confusion pour chaque modèle
- Analyse des erreurs (faux positifs/négatifs)

### EF4 : Interprétabilité ⭐ CRUCIAL
**Techniques obligatoires :**

1. **Feature Importance (basique) :**
   - `model.feature_importances_` pour arbres
   - Permutation Importance (recommandée)

2. **SHAP (avancé, valorisant) :**
   - Explicabilité locale (pourquoi CE client ?)
   - Explicabilité globale (variables importantes)
   - Impact positif/négatif des variables

**Objectif :** Permettre à un responsable CRM de comprendre POURQUOI un client est à risque.

---

## ⚠️ POINTS CRITIQUES À RESPECTER

### 🔴 ERREURS FATALES (invalident le projet)

1. **Data Leakage :**
   - ❌ Appliquer preprocessing sur train + test ensemble
   - ✅ Utiliser **sklearn.Pipeline** ou **ColumnTransformer**
   - ✅ fit() sur train UNIQUEMENT, puis transform() sur test

2. **Déséquilibre des classes ignoré :**
   - Le dataset a ~90% fidèles / ~10% churners
   - ✅ Utiliser `class_weight='balanced'` dans tous les modèles
   - ✅ Utiliser `stratify=y` dans train_test_split
   - ✅ Priorité aux métriques : Recall, F1, PR-AUC
   - ❌ Ne PAS se fier uniquement à l'Accuracy

3. **Pas de Deep Learning :**
   - Obligatoire : MLP (Multi-Layer Perceptron)
   - Peut être fait avec : PyTorch, Keras/TensorFlow, sklearn.MLPClassifier

4. **Moins de 4 modèles comparés :**
   - Non-respect du cahier des charges

5. **Dashboard non fonctionnel :**
   - Livrable obligatoire

---

## 📐 MÉTHODOLOGIE RECOMMANDÉE

### Phase 1 : EDA (Analyse Exploratoire) - EN COURS ✅
**Objectifs :**
1. Comprendre la distribution de chaque variable
2. Identifier les relations avec le churn
3. Détecter outliers, valeurs aberrantes
4. Repérer les corrélations entre features
5. Identifier les variables à fort potentiel prédictif

**Approche structurée :**

#### Pour CHAQUE variable CATÉGORIELLE :
```python
# 1. Distribution générale
df['variable'].value_counts()
df['variable'].value_counts(normalize=True) * 100

# 2. Visualisation
sns.countplot(data=df, x='variable', hue='churn')

# 3. Taux de churn par catégorie (CRUCIAL)
churn_by_cat = df.groupby('variable')['churn'].agg(['mean', 'count'])
churn_by_cat['churn_pct'] = churn_by_cat['mean'] * 100
print(churn_by_cat.sort_values('churn_pct', ascending=False))
```

#### Pour CHAQUE variable NUMÉRIQUE :
```python
# 1. Statistiques descriptives
df['variable'].describe()

# 2. Distribution globale
sns.histplot(data=df, x='variable', bins=30, kde=True)
sns.boxplot(data=df, x='variable')

# 3. Distribution PAR CHURN (⭐ le plus important)
sns.boxplot(data=df, x='churn', y='variable')
sns.histplot(data=df, x='variable', hue='churn', kde=True, alpha=0.6)

# 4. Comparaison statistique
df.groupby('churn')['variable'].describe()

# Médiane par groupe
print(f"Fidèles : {df[df['churn']==0]['variable'].median():.1f}")
print(f"Partis  : {df[df['churn']==1]['variable'].median():.1f}")
```

#### Ordre d'analyse prioritaire :
1. **PRIORITÉ 1 - Variables métier critiques :**
   - `tenure_months` (ancienneté)
   - `payment_failures` (échecs paiement)
   - `last_login_days_ago` (engagement récent)
   - `nps_score` (satisfaction)
   - `contract_type` (engagement contractuel)

2. **PRIORITÉ 2 - Variables comportementales :**
   - `monthly_logins`
   - `weekly_active_days`
   - `usage_growth_rate`
   - `support_tickets`

3. **PRIORITÉ 3 - Variables financières :**
   - `monthly_fee`
   - `total_revenue`
   - `discount_applied`

4. **PRIORITÉ 4 - Variables démographiques :**
   - `gender`, `country`, `customer_segment`

#### Matrice de corrélation finale :
```python
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)

# Focus corrélation avec churn
churn_corr = corr_matrix['churn'].sort_values(ascending=False)
```

### Phase 2 : Preprocessing
1. **Gestion des valeurs manquantes** (si présentes)
2. **Encodage des catégorielles :**
   - OneHotEncoder pour nominales
   - LabelEncoder pour ordinales (si pertinent)
3. **Scaling/Normalisation :**
   - StandardScaler (recommandé pour MLP)
   - MinMaxScaler (alternatif)
4. **Feature Engineering (optionnel mais valorisant) :**
   - Ratios : `support_tickets / tenure_months`
   - Évolutions : `usage_growth_rate`
   - Interactions : combinaisons de features

### Phase 3 : Split Train/Test
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['churn', 'customer_id'])
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # ⭐ CRUCIAL pour déséquilibre
)
```

### Phase 4 : Modélisation Progressive

#### Modèle 1 : Baseline (Régression Logistique)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline_baseline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        class_weight='balanced',  # ⭐ Gestion déséquilibre
        max_iter=1000,
        random_state=42
    ))
])

pipeline_baseline.fit(X_train, y_train)
y_pred = pipeline_baseline.predict(X_test)
```

#### Modèle 2 : Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    ))
])
```

#### Modèle 3 : XGBoost / LightGBM
```python
from xgboost import XGBClassifier

# Calculer scale_pos_weight pour déséquilibre
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

pipeline_xgb = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    ))
])
```

#### Modèle 4 : MLP (Deep Learning) - OBLIGATOIRE
```python
from sklearn.neural_network import MLPClassifier

pipeline_mlp = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    ))
])
```

Ou avec Keras/PyTorch pour plus de contrôle.

### Phase 5 : Évaluation & Comparaison
```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    recall_score
)

# Pour chaque modèle
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

**Tableau comparatif final :**
| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Reg | ... | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... | ... |
| XGBoost | ... | ... | ... | ... | ... |
| MLP | ... | ... | ... | ... | ... |

### Phase 6 : Interprétabilité
```python
# Feature Importance (Random Forest, XGBoost)
importances = model.named_steps['classifier'].feature_importances_
feature_names = preprocessor.get_feature_names_out()

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
feat_imp.head(15).plot(kind='barh')

# Permutation Importance (tous modèles)
from sklearn.inspection import permutation_importance

r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
perm_imp = pd.Series(r.importances_mean, index=feature_names).sort_values(ascending=False)

# SHAP (avancé)
import shap

explainer = shap.TreeExplainer(model.named_steps['classifier'])
shap_values = explainer.shap_values(X_test_transformed)

shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)
```

### Phase 7 : Dashboard (Streamlit)
```python
import streamlit as st

st.title("🎯 Prédiction du Churn Client")

# Saisie utilisateur
col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Ancienneté (mois)", 0, 60, 12)
    monthly_fee = st.number_input("Frais mensuels ($)", 0, 500, 50)
    
with col2:
    payment_failures = st.number_input("Échecs de paiement", 0, 10, 0)
    nps_score = st.slider("NPS Score", 0, 100, 70)

# Prédiction
if st.button("Prédire"):
    input_data = pd.DataFrame({...})
    proba = model.predict_proba(input_data)[0, 1]
    
    st.metric("Probabilité de Churn", f"{proba*100:.1f}%")
    
    if proba > 0.7:
        st.error("⚠️ Client à HAUT RISQUE")
    elif proba > 0.4:
        st.warning("⚠️ Client à risque modéré")
    else:
        st.success("✅ Client stable")
```

### Phase 8 : API (optionnelle)
```python
from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load('models/best_model.pkl')

@app.post("/predict")
def predict(data: dict):
    df_input = pd.DataFrame([data])
    proba = model.predict_proba(df_input)[0, 1]
    return {"churn_probability": float(proba)}

@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## 📁 ARBORESCENCE PROJET RECOMMANDÉE

```
soutenance_m1_data_science/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── doc/
│   ├── Projet M1 DE Sujet 2.md        # Cahier des charges
│   ├── rapport_projet.md              # Rapport final
│   └── presentation.pdf               # Support présentation
│
├── data/
│   ├── raw/
│   │   └── customer_churn_business_dataset.csv
│   ├── processed/
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── external/
│
├── notebooks/
│   ├── 01_EDA.ipynb                   # ✅ EN COURS
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_baseline.ipynb
│   ├── 04_modeling_advanced.ipynb
│   ├── 05_deep_learning.ipynb
│   └── 06_interpretability.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── ensemble.py
│   │   ├── deep_learning.py
│   │   └── model_trainer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── interpretability.py
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── components.py
│   └── api/                          # Optionnel
│       ├── __init__.py
│       └── main.py
│
├── models/                            # Modèles sauvegardés
│   ├── baseline_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── mlp_model.pkl
│   └── best_model.pkl
│
├── outputs/
│   ├── figures/
│   │   ├── eda/
│   │   ├── model_comparison/
│   │   └── interpretability/
│   ├── reports/
│   └── logs/
│
├── scripts/
│   ├── train_all_models.py
│   ├── run_dashboard.sh
│   └── run_api.sh
│
├── tests/                             # Optionnel
│   └── test_preprocessing.py
│
└── config/
    ├── config.yaml
    └── model_params.yaml
```

---

## 📊 ÉTAT D'AVANCEMENT ACTUEL

### ✅ Réalisé
- [x] Compréhension du sujet
- [x] Identification de la tâche prédictive (Classification Churn)
- [x] Chargement du dataset
- [x] Début de l'EDA :
  - Distribution de la variable cible `churn`
  - Identification du déséquilibre de classes (90/10)
  - Calcul de l'impact business (revenu à risque)
  - Identification des 31 variables

### 🔄 En cours
- [ ] **EDA feature par feature** (prochaine étape)
  - Analyse variables catégorielles
  - Analyse variables numériques
  - Relation avec churn pour chaque variable
  - Matrice de corrélation

### ⏳ À faire
- [ ] Preprocessing complet
- [ ] Feature engineering
- [ ] Modélisation (4+ modèles)
- [ ] Évaluation comparée
- [ ] Interprétabilité (SHAP)
- [ ] Dashboard Streamlit
- [ ] API (optionnel)
- [ ] Rapport final
- [ ] Présentation

---

## 🎯 INSIGHTS DÉJÀ IDENTIFIÉS

### Déséquilibre des classes
- **90.2%** clients fidèles (churn = 0)
- **9.8%** clients partis (churn = 1)
- **Ratio :** 9:1 environ

**Conséquences :**
- ✅ Utiliser `class_weight='balanced'`
- ✅ Stratified split obligatoire
- ✅ Métriques prioritaires : Recall, F1, PR-AUC
- ⚠️ Ne pas se fier à l'Accuracy seule

### Impact business (calculé)
- Revenu total généré : ~$XXX,XXX
- Revenu à risque (churners) : ~$XX,XXX
- Pourcentage à risque : ~XX%

---

## 📝 RECOMMANDATIONS MÉTHODOLOGIQUES CLÉS

### R1 : Démarche progressive
1. **Baseline simple** (Régression Logistique)
2. **Modèles plus complexes** (RF, XGBoost)
3. **Deep Learning** (MLP)
4. **Justifier** chaque gain de complexité

### R2 : Éviter le data leakage (IMPÉRATIF)
- Preprocessing uniquement sur train set
- Utiliser sklearn.Pipeline
- Erreur = invalidité totale des résultats

### R3 : Validation croisée
- Utiliser CV pour robustesse
- Moins de dépendance au split unique
- Meilleure comparaison des modèles

### R4 : Analyse des erreurs
- Ne pas se contenter de scores globaux
- Matrices de confusion détaillées
- Étudier les cas mal prédits
- Discuter les raisons possibles

### R5 : Interprétabilité = valeur métier
- Expliquer POURQUOI un client est à risque
- Identifier les leviers d'action
- Transformer modèle statistique en outil décisionnel

### R6 : Code structuré et versionné
- Organiser en modules
- Commits Git réguliers et explicites
- Éviter notebook monolithique
- Documentation claire

### R7 : Posture professionnelle
- Documenter les choix
- Justifier les décisions
- Argumenter les comparaisons
- Démarche scientifique structurée
- Discuter les limites
- Recommandations opérationnelles

---

## 🚀 PROCHAINES ÉTAPES IMMÉDIATES

### 1. Finaliser l'EDA (priorité absolue)
- Analyser TOUTES les variables une par une
- Focus sur relation avec `churn`
- Identifier variables à fort potentiel prédictif
- Matrice de corrélation
- Conclusions EDA documentées

### 2. Preprocessing
- Pipeline sklearn structuré
- Encodage catégorielles
- Scaling numériques
- Train/test split stratifié

### 3. Modélisation baseline
- Régression Logistique
- Évaluation initiale
- Point de référence pour comparaisons

### 4. Modèles avancés
- Random Forest
- XGBoost
- MLP (Deep Learning)
- Comparaison rigoureuse

### 5. Dashboard
- Interface Streamlit
- Prédictions en temps réel
- Visualisations décisionnelles

---

## 💡 IDÉES DE FEATURE ENGINEERING (à explorer pendant EDA)

Potentielles variables dérivées pertinentes :
- `support_tickets / tenure_months` (intensité support)
- `monthly_fee * tenure_months` (LTV estimée)
- `payment_failures / tenure_months` (taux échec)
- Catégorisation de `last_login_days_ago` (actif, inactif, dormant)
- Catégorisation de `tenure_months` (nouveau, établi, fidèle)
- Ratio `usage_growth_rate` < 0 (engagement en baisse)
- Score composite engagement (logins + active_days + session_time)

---

## 🎓 COMPÉTENCES RNCP ÉVALUÉES

### C3.1 - Préparation des données
✅ Transformation et nettoyage  
✅ Qualité optimale  
✅ Documentation des étapes  

### C3.2 - Communication infographique
✅ Dashboard interactif  
✅ Visualisations claires et inclusives  
✅ Extraction de connaissances en temps réel  
✅ Aide à la décision  

### C3.3 - Analyse exploratoire
✅ Techniques statistiques  
✅ Insights exploitables  
✅ Alignement objectifs stratégiques  

### C4.1 - Stratégie d'intégration IA
✅ Cas d'usage pertinents  
✅ Impact sur processus métier  
✅ Feuille de route réalisable  

### C4.2 - Développement modèles prédictifs
✅ Machine Learning  
✅ Prétraitement adapté  
✅ Algorithmes testés et justifiés  
✅ Codes fonctionnels  
✅ Résultats alignés objectifs métier  

### C4.3 - Évaluation performance
✅ Comparaison de plusieurs modèles  
✅ Métriques appropriées  
✅ Améliorations proposées  
✅ Modèle final validé  
✅ Écoresponsabilité  

---

## ⚠️ CHECKLIST AVANT REMISE FINALE

### Code & Technique
- [ ] 4+ modèles implémentés
- [ ] 1 modèle Deep Learning (MLP)
- [ ] Pipeline sklearn sans data leakage
- [ ] Gestion déséquilibre classes
- [ ] Métriques adaptées (Recall, F1, ROC-AUC)
- [ ] Feature Importance / SHAP
- [ ] Code structuré et modulaire
- [ ] Git avec commits réguliers

### Dashboard
- [ ] Streamlit fonctionnel
- [ ] Saisie de scénarios
- [ ] Prédictions en temps réel
- [ ] Visualisations claires
- [ ] Comparaison modèles
- [ ] KPI métier (clients à risque, revenu)

### Documentation
- [ ] Rapport complet et argumenté
- [ ] Choix justifiés
- [ ] Analyse critique
- [ ] Discussion des limites
- [ ] Recommandations opérationnelles
- [ ] Support présentation

### Présentation
- [ ] Démonstration fonctionnelle
- [ ] Tous les membres participent
- [ ] Posture professionnelle
- [ ] Réponses aux questions

---

## 📚 RESSOURCES & RÉFÉRENCES

### Documentation technique
- [Scikit-learn Pipeline](https://scikit-learn.org/stable/modules/compose.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Concepts clés
- Imbalanced Classification
- Pipeline & Data Leakage Prevention
- Model Interpretability (SHAP, Feature Importance)
- Customer Churn Prediction
- Multi-Model Comparison

---

## 🎯 CRITÈRES DE RÉUSSITE ABSOLUS

### Projet validé si :
✅ 4+ modèles comparés rigoureusement  
✅ 1 modèle Deep Learning (MLP) inclus  
✅ Dashboard interactif fonctionnel  
✅ Pas de data leakage  
✅ Déséquilibre des classes géré  
✅ Interprétabilité démontrée  
✅ Rapport professionnel argumenté  
✅ Présentation convaincante avec démonstration  

### Projet excellent si (bonus) :
⭐ API REST fonctionnelle  
⭐ SHAP implémenté (pas seulement Feature Importance)  
⭐ Feature engineering créatif  
⭐ Plusieurs tâches prédictives (churn + revenu à risque)  
⭐ Cross-validation rigoureuse  
⭐ Tests unitaires  
⭐ Déploiement cloud (Streamlit Cloud, Heroku, etc.)  

---

## 📞 QUESTIONS À SE POSER TOUT AU LONG DU PROJET

### Pendant l'EDA :
- Cette variable a-t-elle un lien fort avec le churn ?
- Y a-t-il des différences significatives entre churners et non-churners ?
- Quelles variables semblent les plus prometteuses ?
- Y a-t-il des corrélations fortes à surveiller ?
- Quelles features dérivées pourraient être utiles ?

### Pendant la modélisation :
- Le modèle gère-t-il le déséquilibre des classes ?
- Les métriques sont-elles adaptées au contexte business ?
- Le modèle plus complexe apporte-t-il un gain réel ?
- Y a-t-il du surapprentissage (overfitting) ?
- Le modèle est-il stable (cross-validation) ?

### Pendant l'interprétabilité :
- Pourquoi CE client est-il prédit comme churner ?
- Quelles variables ont le plus d'impact ?
- Les explications sont-elles cohérentes avec la logique métier ?
- Un responsable CRM peut-il actionner ces insights ?

### Pendant le dashboard :
- L'interface est-elle claire pour un non-data scientist ?
- Les prédictions sont-elles expliquées ?
- Les KPI métier sont-ils visibles (revenu à risque, nb clients) ?
- L'utilisateur peut-il simuler des scénarios ?

---

**Document créé le 20 mai 2026**  
**Statut : EDA en cours - Phase d'analyse feature par feature**  
**Version : 1.0**

---

## 🔄 MISE À JOUR SUIVANTE

Une fois l'EDA terminée, mettre à jour ce document avec :
- Liste des variables à fort potentiel prédictif identifiées
- Corrélations significatives découvertes
- Décisions de feature engineering
- Variables à transformer/binning
- Outliers à traiter
