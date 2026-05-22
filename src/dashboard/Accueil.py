"""
Dashboard Streamlit pour la visualisation et prédiction de churn.

Usage:
    streamlit run src/dashboard/app.py
"""

from pathlib import Path
import sys
import joblib
import streamlit as st

BASE_PATH = Path(__file__).resolve().parents[2]

if str(BASE_PATH) not in sys.path:
    sys.path.append(str(BASE_PATH))

from src.chargement_donnees import charger_donnees
from src.traitement_null_outliers import traiter_donnees
from src.commun import variables_numeriques, variables_categorielles
import src.model_xgboost as model_xgboost

MODELS_PATH = BASE_PATH / "models"
SEUIL_CHURN = 0.32

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

# ── En-tête ──────────────────────────────────────────────────────────────────
st.title("📊 Churn Intelligence - Système de Rétention Client")
st.markdown("""
Bienvenue sur la plateforme de prédiction du churn client.

Utilisez le menu à gauche pour naviguer entre les pages :
- **Accueil** : KPIs clés et vue globale ;
- **Prédiction** : estimez la probabilité de churn d'un client.
""")
st.markdown("---")


# ── Chargement des données et prédictions (mis en cache) ─────────────────────
@st.cache_data
def charger_et_predire():
    df = charger_donnees()
    df = traiter_donnees(df)

    preprocessor = joblib.load(MODELS_PATH / "preprocessor.pkl")
    model = model_xgboost.load()

    X = df[variables_numeriques + variables_categorielles]
    X_transformed = preprocessor.transform(X)

    probas = model.predict_proba(X_transformed)[:, 1]
    df = df.copy()
    df["proba_churn"] = probas
    return df


df = charger_et_predire()

nb_clients = len(df)
nb_a_risque = int((df["proba_churn"] >= SEUIL_CHURN).sum())
revenu_a_risque = df.loc[df["proba_churn"] >= SEUIL_CHURN, "monthly_fee"].sum()
taux_churn = nb_a_risque / nb_clients * 100

st.subheader("Détail des KPIs")

# with st.expander("Détails des KPIs", expanded=True):

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="👥 Clients total",
    value=f"{nb_clients:,}",
)
col2.metric(
    label="⚠️ Clients à risque",
    value=f"{nb_a_risque:,}",
    help=f"Seuil de probabilité de churn ≥ {SEUIL_CHURN:.0%}",
)
col3.metric(
    label="📉 Taux de désabonnement estimé",
    value=f"{taux_churn:.1f}%",
)
col4.metric(
    label="💰 Revenu mensuel à risque",
    value=f"${revenu_a_risque:,.0f}",
    help="Somme des frais mensuels des clients à risque",
)

st.markdown("---")
st.subheader("Facteurs de risque de résiliation")

st.image(BASE_PATH / "img" / "features_importance.png")
