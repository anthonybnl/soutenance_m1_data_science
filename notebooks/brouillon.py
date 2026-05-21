# =============================================================================
# BROUILLON — preprocess.ipynb
# Copier chaque bloc dans une cellule du notebook.
# Les blocs "# MARKDOWN" sont à mettre en cellule Markdown (sans le #).
# =============================================================================


# =============================================================================
# MARKDOWN — Titre du notebook
# =============================================================================
"""
# 🔧 Preprocessing — Préparation des Données

> **Projet M1 Data Science** - Expert en Ingénierie de Données (RNCP40875 - Bloc 2)

Ce notebook couvre l'intégralité du pipeline de preprocessing :
nettoyage, feature engineering, encodage, scaling, et sauvegarde des données
prêtes pour la modélisation.

**Anti data-leakage** : toutes les transformations sont fittées **uniquement
sur le train set**, puis appliquées sur train et test séparément via `sklearn.Pipeline`.

---

| Section | Contenu |
|---|---|
| 0 | Setup & Imports |
| 1 | Chargement & vérification rapide |
| 2 | Suppression des colonnes inutiles |
| 3 | Traitement des valeurs manquantes |
| 4 | Feature Engineering |
| 5 | Séparation X / y |
| 6 | Train / Test Split stratifié |
| 7 | Pipeline ColumnTransformer |
| 8 | Fit sur train — Transform sur train + test |
| 9 | Vérification post-processing |
| 10 | Sauvegarde |
"""


# =============================================================================
# MARKDOWN — Section 0
# =============================================================================
"""
---
## Section 0 — Setup & Imports
"""


# =============================================================================
# CODE — Cellule 0.1 : imports
# =============================================================================
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

RANDOM_STATE = 42
TEST_SIZE    = 0.2


# =============================================================================
# CODE — Cellule 0.2 : chemins
# =============================================================================
BASE_PATH      = Path.cwd().parent
DATA_RAW_PATH  = BASE_PATH / 'data' / 'customer_churn_business_dataset.csv'
DATA_PROC_PATH = BASE_PATH / 'data' / 'processed'
MODELS_PATH    = BASE_PATH / 'models'

DATA_PROC_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

print(f'Données brutes   : {DATA_RAW_PATH}')
print(f'Données traitées : {DATA_PROC_PATH}')
print(f'Modèles          : {MODELS_PATH}')


# =============================================================================
# MARKDOWN — Section 1
# =============================================================================
"""
---
## Section 1 — Chargement & Vérification Rapide

Sanity check avant de travailler : shape, aperçu, types, valeurs manquantes, doublons.
L'EDA approfondie a déjà été faite dans `01_EDA.ipynb`.
"""


# =============================================================================
# CODE — Cellule 1.1 : chargement + aperçu
# =============================================================================
df = pd.read_csv(DATA_RAW_PATH)

print(df.shape)
df.head()


# =============================================================================
# CODE — Cellule 1.2 : types et valeurs manquantes par colonne
# =============================================================================
df.info()


# =============================================================================
# CODE — Cellule 1.3 : valeurs manquantes par colonne
# =============================================================================
df.isna().sum()


# =============================================================================
# CODE — Cellule 1.4 : doublons
# =============================================================================
df.duplicated().sum()


# =============================================================================
# CODE — Cellule 1.5 : distribution de la cible
# =============================================================================
df['churn'].value_counts()


# =============================================================================
# CODE — Cellule 1.6 : distribution en pourcentage
# =============================================================================
df['churn'].value_counts(normalize=True).mul(100).round(2)


# =============================================================================
# MARKDOWN — Section 2
# =============================================================================
"""
---
## Section 2 — Suppression des colonnes inutiles

- `customer_id` : identifiant unique, aucune valeur prédictive → supprimé
- `city` / `country` : trop de bruit géographique pour un modèle de churn SaaS,
  faible valeur prédictive attendue → supprimés
"""


# =============================================================================
# CODE — Cellule 2.1 : suppression
# =============================================================================
cols_to_drop = ['customer_id', 'city', 'country']

df = df.drop(columns=cols_to_drop)

print(df.shape)
df.head(3)


# =============================================================================
# CODE — Cellule 2.2 : vérification
# =============================================================================
df.columns.tolist()


# =============================================================================
# MARKDOWN — Section 3
# =============================================================================
"""
---
## Section 3 — Traitement des Valeurs Manquantes

Seule colonne avec des NaN : `complaint_type` (2 045 manquants, soit ~20%).

Ces NaN ne sont **pas** des données inconnues : ils correspondent à des clients
qui n'ont **jamais déposé de plainte**. On remplace donc par la catégorie
`"No_Complaint"` (imputation sémantique, pas statistique).

> Imputer par le mode serait une erreur : ça supposerait que ces clients
> ont eu une plainte, ce qui est faux.
"""


# =============================================================================
# CODE — Cellule 3.1 : vérification avant imputation
# =============================================================================
print("Valeurs manquantes avant :")
print(df['complaint_type'].isna().sum())
print()
print("Valeurs existantes :")
print(df['complaint_type'].value_counts())


# =============================================================================
# CODE — Cellule 3.2 : imputation sémantique
# =============================================================================
df['complaint_type'] = df['complaint_type'].fillna('No_Complaint')


# =============================================================================
# CODE — Cellule 3.3 : vérification après imputation
# =============================================================================
print("Valeurs manquantes après :")
print(df['complaint_type'].isna().sum())
print()
print("Distribution complaint_type :")
print(df['complaint_type'].value_counts())


# =============================================================================
# CODE — Cellule 3.4 : confirmation — plus aucun NaN dans le dataset
# =============================================================================
df.isna().sum()


# =============================================================================
# MARKDOWN — Section 4
# =============================================================================
"""
---
## Section 4 — Encodage & Normalisation

**Règle anti data-leakage** : le preprocessor est fitté **uniquement sur X_train**,
puis appliqué sur X_test. Aucune information du test n'influence les transformations.
"""


# =============================================================================
# CODE — Cellule 4.1 : séparation X / y
# =============================================================================
X = df.drop(columns=['churn'])
y = df['churn']

print(X.shape)
print(y.value_counts())


# =============================================================================
# CODE — Cellule 4.2 : train / test split stratifié
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print()
print("Distribution churn — train :")
print(y_train.value_counts(normalize=True).mul(100).round(2))
print()
print("Distribution churn — test :")
print(y_test.value_counts(normalize=True).mul(100).round(2))


# =============================================================================
# CODE — Cellule 4.3 : définition des groupes de colonnes
# =============================================================================
# 2 valeurs → OrdinalEncoder (0/1), pas besoin de créer 2 colonnes
binary_cols = ['gender', 'discount_applied', 'price_increase_last_3m']

# 3+ valeurs → OneHotEncoder
nominal_cols = [
    'customer_segment',
    'signup_channel',
    'contract_type',
    'payment_method',
    'complaint_type',
    'survey_response',
]

# Numériques → StandardScaler
numerical_cols = [col for col in X.columns if col not in binary_cols + nominal_cols]

print("Binaires  :", binary_cols)
print("Nominales :", nominal_cols)
print("Numériques:", numerical_cols)


# =============================================================================
# CODE — Cellule 4.4 : construction du ColumnTransformer
# =============================================================================
preprocessor = ColumnTransformer(transformers=[
    ('bin', OrdinalEncoder(), binary_cols),
    ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), nominal_cols),
    ('num', StandardScaler(), numerical_cols),
])


# =============================================================================
# CODE — Cellule 4.5 : fit sur train → transform train + test
# =============================================================================
X_train_processed = preprocessor.fit_transform(X_train)  # apprend ET transforme
X_test_processed  = preprocessor.transform(X_test)        # transforme seulement

print(f"X_train_processed : {X_train_processed.shape}")
print(f"X_test_processed  : {X_test_processed.shape}")


# =============================================================================
# CODE — Cellule 4.6 : noms des features après encodage
# =============================================================================
feature_names = preprocessor.get_feature_names_out()
print(f"Nombre de features : {len(feature_names)}")
print(feature_names)


# =============================================================================
# CODE — Cellule 4.7 : vérification — zéro NaN
# =============================================================================
print("NaN dans X_train_processed :", np.isnan(X_train_processed).sum())
print("NaN dans X_test_processed  :", np.isnan(X_test_processed).sum())


# =============================================================================
# ============================================================================
# 03_ML.ipynb
# ============================================================================
# =============================================================================


# =============================================================================
# MARKDOWN — Titre
# =============================================================================
"""
# 🤖 Modélisation — Classification du Churn

Entraînement et évaluation de 4 modèles de classification :
1. Régression Logistique (baseline)
2. Random Forest
3. XGBoost
4. MLP (Deep Learning)

**Données** : chargées depuis `data/processed/` (produites par `02_Preprocess.ipynb`)
"""


# =============================================================================
# MARKDOWN — Section 0
# =============================================================================
"""
---
## Section 0 — Setup & Chargement des données
"""


# =============================================================================
# CODE — Cellule 0.1 : imports
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


# =============================================================================
# CODE — Cellule 0.2 : chargement des données
# =============================================================================
DATA_PROC_PATH = Path("../data/processed")

X_train = np.load(DATA_PROC_PATH / "X_train.npy")
X_test  = np.load(DATA_PROC_PATH / "X_test.npy")
y_train = np.load(DATA_PROC_PATH / "y_train.npy")
y_test  = np.load(DATA_PROC_PATH / "y_test.npy")

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : {y_train.shape}  — churn rate : {y_train.mean()*100:.1f}%")
print(f"y_test  : {y_test.shape}   — churn rate : {y_test.mean()*100:.1f}%")


# =============================================================================
# CODE — Cellule 0.3 : fonction d'évaluation réutilisable
# =============================================================================
def evaluate_model(model, X_test, y_test, model_name="Modèle"):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"=== {model_name} ===")
    print(classification_report(y_test, y_pred, target_names=["Fidèle", "Churné"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")

    # Matrice de confusion
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Fidèle", "Churné"],
        ax=axes[0], colorbar=False
    )
    axes[0].set_title(f"Matrice de confusion — {model_name}")

    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
    axes[1].set_title(f"Courbe ROC — {model_name}")

    plt.tight_layout()
    plt.show()

    return {
        "modele"  : model_name,
        "roc_auc" : round(roc_auc_score(y_test, y_proba), 4),
    }


# =============================================================================
# MARKDOWN — Section 1
# =============================================================================
"""
---
## Section 1 — Régression Logistique (Baseline)

Premier modèle de référence. Simple et interprétable.
`class_weight='balanced'` pour compenser le déséquilibre 90/10.
"""


# =============================================================================
# CODE — Cellule 1.1 : entraînement
# =============================================================================
lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train)
print("Entraînement terminé.")


# =============================================================================
# CODE — Cellule 1.2 : évaluation
# =============================================================================
results_lr = evaluate_model(lr, X_test, y_test, "Régression Logistique")
