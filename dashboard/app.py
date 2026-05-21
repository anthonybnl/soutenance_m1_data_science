import streamlit as st

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("Churn Intelligence — Système de Rétention Client")
st.markdown("""
Bienvenue sur la plateforme de prédiction du churn client.

Utilisez le menu à gauche pour naviguer entre les pages :
- **Prédiction** : estimez la probabilité de churn d'un client
- **Analyse** : explorez les données et les indicateurs clés
""")
