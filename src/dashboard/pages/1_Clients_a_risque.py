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

st.title("📊 Churn Intelligence - Clients à risque")
st.markdown("---")


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

# nb_a_risque = int((df["proba_churn"] >= SEUIL_CHURN).sum())
# revenu_a_risque = df.loc[df["proba_churn"] >= SEUIL_CHURN, "monthly_fee"].sum()
# taux_churn = nb_a_risque / nb_clients * 100

df["a_risque"] = df["proba_churn"] >= SEUIL_CHURN

# Revenu attendu à risque = monthly_fee pondéré par la probabilité de churn
df["revenu_risque"] = df["monthly_fee"] * df["proba_churn"]

# ── Top 10 clients à fidéliser en priorité ────────────────────────────────────
st.subheader("🎯 Top 10 clients à fidéliser en priorité")
st.caption(
    "Classement par **revenu mensuel à risque** (frais mensuels × probabilité de churn). "
    "Un client avec une forte mensualité et une probabilité élevée de résiliation représente "
    "la perte financière la plus critique."
)

COLONNES_AFFICHAGE = {
    "customer_id": "ID Client",
    "csat_score": "Score de satisfaction",
    "payment_failures": "Échecs paiement",
    "tenure_months": "Ancienneté (mois)",
    "monthly_logins": "Connexions mensuelles",
    "customer_segment": "Segment",
    "monthly_fee": "Frais mensuels ($)",
    "proba_churn": "Probabilité de churn",
    "revenu_risque": "Revenu à risque ($)",
}

top10 = (
    df[df["a_risque"]]
    .sort_values("revenu_risque", ascending=False)
    .head(10)[list(COLONNES_AFFICHAGE.keys())]
    .rename(columns=COLONNES_AFFICHAGE)
    .reset_index(drop=True)
)
top10.index += 1  # classement 1-10

st.dataframe(
    top10.style.format(
        {
            "Frais mensuels ($)": "${:.2f}",
            "Probabilité de churn": "{:.1%}",
            "Revenu à risque ($)": "${:.2f}",
        }
    )
    .background_gradient(subset=["Revenu à risque ($)"], cmap="Reds")
    .background_gradient(subset=["Probabilité de churn"], cmap="Oranges"),
    width='stretch',
)
