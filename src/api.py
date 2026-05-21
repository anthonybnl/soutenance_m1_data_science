from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

BASE_PATH = Path(__file__).resolve().parents[1]
MODELS_PATH = BASE_PATH / "models"

MODEL_PATH = MODELS_PATH / "xgboost.pkl"
PREPROCESSOR_PATH = MODELS_PATH / "preprocessor.pkl"

# ---------------------------------------------------------------------------
# Chargement du modèle et du preprocessor au démarrage
# ---------------------------------------------------------------------------

def _load_artifact(path: Path, nom: str):
    if not path.exists():
        raise RuntimeError(
            f"{nom} introuvable : {path}\n"
            "Lance d'abord : python -m src.model_xgboost"
        )
    return joblib.load(path)


model = _load_artifact(MODEL_PATH, "Modèle XGBoost")
preprocessor = _load_artifact(PREPROCESSOR_PATH, "Preprocessor")

# ---------------------------------------------------------------------------
# Schémas Pydantic
# ---------------------------------------------------------------------------

class ClientInput(BaseModel):
    # Infos démographiques
    age: int = Field(..., ge=0, le=120, example=35)
    gender: str = Field(..., example="Male")
    city: str = Field(..., example="Paris")

    # Abonnement
    tenure_months: int = Field(..., ge=0, example=24)
    customer_segment: str = Field(..., example="SME")
    signup_channel: str = Field(..., example="Web")
    contract_type: str = Field(..., example="Monthly")
    monthly_fee: float = Field(..., ge=0, example=49.0)
    total_revenue: float = Field(..., ge=0, example=1176.0)
    payment_method: str = Field(..., example="Credit Card")
    discount_applied: str = Field(..., example="No")
    price_increase_last_3m: str = Field(..., example="Yes")

    # Comportement
    monthly_logins: int = Field(..., ge=0, example=12)
    weekly_active_days: int = Field(..., ge=0, le=7, example=4)
    avg_session_time: float = Field(..., ge=0, example=20.5)
    features_used: int = Field(..., ge=0, example=5)
    usage_growth_rate: float = Field(..., example=0.05)
    last_login_days_ago: int = Field(..., ge=0, example=3)

    # Paiement & support
    payment_failures: int = Field(..., ge=0, example=0)
    support_tickets: int = Field(..., ge=0, example=1)
    avg_resolution_time: float = Field(..., ge=0, example=24.0)
    complaint_type: Optional[str] = Field(None, example="Technical")

    # Satisfaction
    csat_score: float = Field(..., ge=0, le=5, example=3.5)
    escalations: int = Field(..., ge=0, example=0)
    nps_score: int = Field(..., ge=-100, le=100, example=20)
    survey_response: str = Field(..., example="Neutral")

    # Marketing
    email_open_rate: float = Field(..., ge=0, le=1, example=0.45)
    marketing_click_rate: float = Field(..., ge=0, le=1, example=0.20)
    referral_count: int = Field(..., ge=0, example=1)


class PredictionOutput(BaseModel):
    churn: int
    proba_churn: float
    risque: str


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Churn Prediction API",
    description="Prédit la probabilité de churn d'un client via XGBoost.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost"}


@app.post("/predict", response_model=PredictionOutput)
def predict(client: ClientInput):
    data = client.model_dump()

    # complaint_type manquant → No_Complaint (cohérent avec le preprocessing)
    if data["complaint_type"] is None:
        data["complaint_type"] = "No_Complaint"

    df = pd.DataFrame([data])

    try:
        X = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur de preprocessing : {e}")

    proba = float(model.predict_proba(X)[0, 1])
    prediction = int(proba >= 0.5)

    if proba < 0.1:
        risque = "Faible"
    elif proba < 0.4:
        risque = "Modéré"
    else:
        risque = "Élevé"

    return PredictionOutput(churn=prediction, proba_churn=round(proba, 4), risque=risque)
