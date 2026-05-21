from pathlib import Path
import sys
import joblib
import streamlit as st
import plotly.express as px

BASE_PATH = Path(__file__).resolve().parents[3]

if str(BASE_PATH) not in sys.path:
    sys.path.append(str(BASE_PATH))

from src.chargement_donnees import charger_donnees
from src.traitement_null_outliers import traiter_donnees
from src.commun import variables_numeriques, variables_categorielles
import src.model_xgboost as model_xgboost

MODELS_PATH = BASE_PATH / "models"
SEUIL_CHURN = 0.5

st.set_page_config(
    page_title="Churn Intelligence - Données",
    page_icon="📊",
    layout="wide",
)

st.title("📈 Churn Intelligence - Analyse des données")
st.markdown("Visualisation des facteurs clés liés au risque de résiliation.")
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
df["statut"] = df["proba_churn"].apply(
    lambda p: "À risque" if p >= SEUIL_CHURN else "Fidèle"
)

# ── 1. Taux de résiliation par segment ───────────────────────────────────────
taux_segment = (
    df.groupby("customer_segment")["proba_churn"]
    .apply(lambda x: (x >= SEUIL_CHURN).mean() * 100)
    .reset_index()
    .rename(columns={"proba_churn": "taux_churn"})
    .sort_values("taux_churn", ascending=False)
)
fig = px.bar(
    taux_segment,
    x="customer_segment",
    y="taux_churn",
    title="Taux de résiliation par segment",
    labels={"customer_segment": "Segment", "taux_churn": "Taux de résiliation (%)"},
    color="taux_churn",
    color_continuous_scale="Reds",
    text=taux_segment["taux_churn"].apply(lambda v: f"{v:.1f}%"),
)
fig.update_traces(textposition="outside")
fig.update_layout(
    showlegend=False, coloraxis_showscale=False, height=400, margin=dict(t=40, b=10)
)
st.plotly_chart(fig, width="content")

st.markdown("---")

# ── 2. Taux de résiliation par type de contrat ────────────────────────────────
taux_contrat = (
    df.groupby("contract_type")["proba_churn"]
    .apply(lambda x: (x >= SEUIL_CHURN).mean() * 100)
    .reset_index()
    .rename(columns={"proba_churn": "taux_churn"})
    .sort_values("taux_churn", ascending=False)
)
fig = px.bar(
    taux_contrat,
    x="contract_type",
    y="taux_churn",
    title="Taux de résiliation par type de contrat",
    labels={"contract_type": "Contrat", "taux_churn": "Taux de résiliation (%)"},
    color="taux_churn",
    color_continuous_scale="Oranges",
    text=taux_contrat["taux_churn"].apply(lambda v: f"{v:.1f}%"),
)
fig.update_traces(textposition="outside")
fig.update_layout(
    showlegend=False, coloraxis_showscale=False, height=400, margin=dict(t=40, b=10)
)
st.plotly_chart(fig, width="content")

st.markdown("---")

# ── 3. Taux de résiliation selon les échecs de paiement ──────────────────────
taux_paiement = (
    df.groupby("payment_failures")["proba_churn"]
    .apply(lambda x: (x >= SEUIL_CHURN).mean() * 100)
    .reset_index()
    .rename(columns={"proba_churn": "taux_churn"})
)
fig = px.bar(
    taux_paiement,
    x="payment_failures",
    y="taux_churn",
    title="Taux de résiliation selon les échecs de paiement",
    labels={
        "payment_failures": "Nb échecs de paiement",
        "taux_churn": "Taux de résiliation (%)",
    },
    color="taux_churn",
    color_continuous_scale="RdYlGn_r",
    text=taux_paiement["taux_churn"].apply(lambda v: f"{v:.1f}%"),
)
fig.update_traces(textposition="outside")
fig.update_layout(
    showlegend=False, coloraxis_showscale=False, height=400, margin=dict(t=40, b=10)
)
st.plotly_chart(fig, width="content")

st.markdown("---")

# ── 4. Distribution du score de satisfaction (CSAT) ──────────────────────────
fig = px.histogram(
    df,
    x="csat_score",
    color="statut",
    barmode="overlay",
    opacity=0.65,
    nbins=20,
    title="Distribution du score de satisfaction (CSAT)",
    labels={
        "csat_score": "Score CSAT",
        "count": "Nombre de clients",
        "statut": "Statut",
    },
    color_discrete_map={"À risque": "#ef4444", "Fidèle": "#60a5fa"},
)
fig.update_layout(
    height=400, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.2)
)
st.plotly_chart(fig, width="content")

st.markdown("---")

# ── 5. Distribution de l'ancienneté client ────────────────────────────────────
fig = px.histogram(
    df,
    x="tenure_months",
    color="statut",
    barmode="overlay",
    opacity=0.65,
    nbins=30,
    title="Distribution de l'ancienneté client",
    labels={
        "tenure_months": "Ancienneté (mois)",
        "count": "Nombre de clients",
        "statut": "Statut",
    },
    color_discrete_map={"À risque": "#ef4444", "Fidèle": "#60a5fa"},
)
fig.update_layout(
    height=400, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.2)
)
st.plotly_chart(fig, width="content")

st.markdown("---")

# ── 6. Distribution des connexions mensuelles ────────────────────────────────
fig = px.histogram(
    df,
    x="monthly_logins",
    color="statut",
    barmode="overlay",
    opacity=0.65,
    nbins=25,
    title="Distribution des connexions mensuelles",
    labels={
        "monthly_logins": "Connexions / mois",
        "count": "Nombre de clients",
        "statut": "Statut",
    },
    color_discrete_map={"À risque": "#ef4444", "Fidèle": "#60a5fa"},
)
fig.update_layout(
    height=400, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.2)
)
st.plotly_chart(fig, width="content")
