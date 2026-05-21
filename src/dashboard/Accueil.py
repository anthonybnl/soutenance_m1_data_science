"""
Dashboard Streamlit pour la visualisation et prédiction de churn.

Usage:
    streamlit run src/dashboard/app.py
"""

from pathlib import Path
import sys
import streamlit as st

BASE_PATH = Path(__file__).resolve().parents[2]

if not str(BASE_PATH) in sys.path:
    sys.path.append(str(BASE_PATH))

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Churn Intelligence — Système de Rétention Client")
st.markdown("""
Bienvenue sur la plateforme de prédiction du churn client.

Utilisez le menu à gauche pour naviguer entre les pages :
- **Accueil** : page d'accueil avec les KPIs clés ;
- **Prédiction** : estimez la probabilité de churn d'un client.
""")

