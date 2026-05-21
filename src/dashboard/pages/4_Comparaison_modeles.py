import requests
import streamlit as st
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[3]

st.set_page_config(
    page_title="Churn Intelligence - Comparaison modèles", page_icon="🔮", layout="wide"
)
st.title("🔮 Churn Intelligence - Comparaison des modèles")

st.markdown("---")

st.subheader("Modèles Machine Learning : Régression Logistique, Random Forest, XGBoost")

st.markdown("""
La matrice de confusion montre les vrais positifs, faux positifs, vrais négatifs et faux négatifs pour chaque modèle, ce qui permet d'évaluer leur précision et leur capacité à éviter les erreurs de classification.
""")

st.image(
    BASE_PATH / "img" / "models_confusion_matrices.png",
    caption="Matrice de confusion des différents modèles",
)

st.markdown("---")

st.markdown("""
La courbe ROC_AUC compare la performance des modèles en termes de taux de vrais positifs et de taux de faux positifs, avec une AUC plus élevée indiquant une meilleure capacité à distinguer les clients à risque.
""")

st.image(
    BASE_PATH / "img" / "models_roc_comparison.png",
    caption="Comparaison des courbes ROC des différents modèles",
)

st.markdown("---")

st.subheader("Modèle Deep Learning : Réseau de Neurones")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""Matrice de confusion du modèle de réseau de neurones""")
    st.image(BASE_PATH / "img" / "mlp_confusion_matrix.png")


with col2:
    st.markdown("""Courbe ROC du modèle de réseau de neurones""")
    st.image(BASE_PATH / "img" / "mlp_roc_curve.png")
