from pathlib import Path

import joblib
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from src.chargement_donnees_preprocessed import charger_donnees_preprocessed

MODELS_PATH = Path(__file__).resolve().parents[1] / "models"


def train():
    X_train, X_test, y_train, y_test = charger_donnees_preprocessed()

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=17,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== XGBoost ===")
    print(classification_report(y_test, y_pred, target_names=["Non-churn", "Churn"]))
    print(f"AUC : {roc_auc_score(y_test, y_proba):.4f}")

    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_PATH / "xgboost.pkl")
    print(f"Modèle sauvegardé : {MODELS_PATH / 'xgboost.pkl'}")

    return model


def load():
    return joblib.load(MODELS_PATH / "xgboost.pkl")


def predict(X):
    model = load()
    return {
        "prediction": model.predict(X).tolist(),
        "proba_churn": model.predict_proba(X)[:, 1].tolist(),
    }


if __name__ == "__main__":
    train()
