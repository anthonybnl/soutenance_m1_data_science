from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from src.chargement_donnees_preprocessed import charger_donnees_preprocessed

MODELS_PATH = Path(__file__).resolve().parents[1] / "models"


def train():
    X_train, X_test, y_train, y_test = charger_donnees_preprocessed()

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== Régression Logistique ===")
    print(classification_report(y_test, y_pred, target_names=["Non-churn", "Churn"]))
    print(f"AUC : {roc_auc_score(y_test, y_proba):.4f}")

    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_PATH / "logistic_regression.pkl")
    print(f"Modèle sauvegardé : {MODELS_PATH / 'logistic_regression.pkl'}")

    return model


def load():
    return joblib.load(MODELS_PATH / "logistic_regression.pkl")


def predict(X):
    model = load()
    return {
        "prediction": model.predict(X).tolist(),
        "proba_churn": model.predict_proba(X)[:, 1].tolist(),
    }


if __name__ == "__main__":
    train()
