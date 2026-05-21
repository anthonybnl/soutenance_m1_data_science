import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Churn Intelligence - Prédiction", page_icon="🔮", layout="wide")
st.title("🔮 Churn Intelligence - Prédiction de la résiliation")
st.markdown("Renseignez les informations du client pour estimer son risque de résiliation.")

# ---------------------------------------------------------------------------
# Vérification santé API
# ---------------------------------------------------------------------------
try:
    health = requests.get(f"{API_URL}/health", timeout=3)
    if health.status_code == 200:
        st.success("API connectée", icon="✅")
    else:
        st.error("API inaccessible — lancez : `uvicorn src.api:app --reload`")
        st.stop()
except requests.exceptions.ConnectionError:
    st.error("API inaccessible — lancez : `uvicorn src.api:app --reload`")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Formulaire client
# ---------------------------------------------------------------------------
with st.form("formulaire_client"):

    st.subheader("Profil du client")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Âge", 18, 80, 35)
        gender = st.selectbox("Genre", ["Male", "Female"])
        city = st.selectbox("Ville", ["Berlin", "Delhi", "Dhaka", "London", "New York", "Sydney", "Toronto"])
        customer_segment = st.selectbox("Segment", ["Individual", "SME", "Enterprise"])

    with col2:
        tenure_months = st.number_input("Ancienneté (mois)", 0, 120, 24)
        contract_type = st.selectbox("Type de contrat", ["Monthly", "Yearly"])
        signup_channel = st.selectbox("Canal d'inscription", ["Web", "Mobile", "Referral"])

    with col3:
        monthly_fee = st.number_input("Frais mensuels ($)", 0, 500, 49)
        total_revenue = st.number_input("Revenu total ($)", 0, 10000, 1176)
        payment_method = st.selectbox("Méthode de paiement", ["Credit Card", "PayPal", "Bank Transfer"])

    st.subheader("Comportement")
    col4, col5, col6 = st.columns(3)

    with col4:
        monthly_logins = st.number_input("Connexions mensuelles", 0, 100, 12)
        weekly_active_days = st.slider("Jours actifs / semaine", 0, 7, 4)
        avg_session_time = st.number_input("Durée moyenne session (min)", 0.0, 120.0, 20.0)

    with col5:
        features_used = st.number_input("Fonctionnalités utilisées", 0, 20, 5)
        usage_growth_rate = st.number_input("Taux de croissance usage", -1.0, 5.0, 0.05, step=0.01)
        last_login_days_ago = st.number_input("Dernière connexion (jours)", 0, 365, 3)

    with col6:
        discount_applied = st.selectbox("Réduction appliquée", ["No", "Yes"])
        price_increase_last_3m = st.selectbox("Hausse de prix (3 mois)", ["No", "Yes"])

    st.subheader("Support & Satisfaction")
    col7, col8, col9 = st.columns(3)

    with col7:
        payment_failures = st.number_input("Échecs de paiement", 0, 20, 0)
        support_tickets = st.number_input("Tickets support", 0, 20, 1)
        avg_resolution_time = st.number_input("Temps résolution moyen (h)", 0.0, 72.0, 24.0)

    with col8:
        complaint_type = st.selectbox(
            "Type de plainte", ["No_Complaint", "Technical", "Billing", "Service"]
        )
        csat_score = st.slider("Score satisfaction (CSAT)", 0.0, 5.0, 3.5, step=0.5)
        escalations = st.number_input("Escalades", 0, 10, 0)

    with col9:
        nps_score = st.slider("NPS Score", -100, 100, 20)
        survey_response = st.selectbox("Réponse enquête", ["Satisfied", "Neutral", "Unsatisfied"])
        email_open_rate = st.slider("Taux ouverture email", 0.0, 1.0, 0.45, step=0.01)
        marketing_click_rate = st.slider("Taux clic marketing", 0.0, 1.0, 0.20, step=0.01)
        referral_count = st.number_input("Parrainages", 0, 20, 1)

    submitted = st.form_submit_button("Prédire le risque de churn", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Appel API et affichage du résultat
# ---------------------------------------------------------------------------
if submitted:
    payload = {
        "age": age,
        "gender": gender,
        "city": city,
        "tenure_months": int(tenure_months),
        "customer_segment": customer_segment,
        "signup_channel": signup_channel,
        "contract_type": contract_type,
        "monthly_fee": float(monthly_fee),
        "total_revenue": float(total_revenue),
        "payment_method": payment_method,
        "discount_applied": discount_applied,
        "price_increase_last_3m": price_increase_last_3m,
        "monthly_logins": int(monthly_logins),
        "weekly_active_days": int(weekly_active_days),
        "avg_session_time": float(avg_session_time),
        "features_used": int(features_used),
        "usage_growth_rate": float(usage_growth_rate),
        "last_login_days_ago": int(last_login_days_ago),
        "payment_failures": int(payment_failures),
        "support_tickets": int(support_tickets),
        "avg_resolution_time": float(avg_resolution_time),
        "complaint_type": complaint_type if complaint_type != "No_Complaint" else None,
        "csat_score": float(csat_score),
        "escalations": int(escalations),
        "nps_score": int(nps_score),
        "survey_response": survey_response,
        "email_open_rate": float(email_open_rate),
        "marketing_click_rate": float(marketing_click_rate),
        "referral_count": int(referral_count),
    }

    with st.spinner("Analyse en cours..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.HTTPError as e:
            st.error(f"Erreur API : {e.response.text if e.response else e}")
            st.stop()
        except Exception as e:
            st.error(f"Erreur inattendue : {e}")
            st.stop()

    st.divider()
    st.subheader("Résultat de l'analyse")

    proba = result["proba_churn"]
    risque = result["risque"]
    churn = result["churn"]

    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        st.metric("Probabilité de churn", f"{proba * 100:.1f}%")

    with col_res2:
        couleur = {"Faible": "green", "Modéré": "orange", "Élevé": "red"}[risque]
        st.markdown(
            f"<h3 style='color:{couleur}'>Niveau de risque : {risque}</h3>",
            unsafe_allow_html=True,
        )

    with col_res3:
        if churn == 1:
            st.error("Ce client est susceptible de résilier.")
        else:
            st.success("Ce client devrait rester fidèle.")

    # Jauge visuelle
    st.progress(proba, text=f"Probabilité de churn : {proba * 100:.1f}%")

    # Recommandation métier
    st.subheader("Recommandation")
    if risque == "Élevé":
        st.warning(
            "Action immédiate recommandée : contacter le client, proposer une offre de fidélisation ou une réduction ciblée."
        )
    elif risque == "Modéré":
        st.info(
            "Surveiller ce client : envoyer un email de satisfaction, vérifier les tickets support récents."
        )
    else:
        st.success("Faible risque : aucune action urgente. Maintenir l'engagement habituel.")
