from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.chargement_donnees_preprocessed import charger_donnees_preprocessed

MODELS_PATH = Path(__file__).resolve().parents[1] / "models"


def train():
    X_train, X_test, y_train, y_test = charger_donnees_preprocessed()

    model = RandomForestClassifier(
        class_weight="balanced", n_estimators=100, random_state=17
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== Random Forest ===")
    print(classification_report(y_test, y_pred, target_names=["Non-churn", "Churn"]))
    print(f"AUC : {roc_auc_score(y_test, y_proba):.4f}")

    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_PATH / "random_forest.pkl")
    print(f"Modèle sauvegardé : {MODELS_PATH / 'random_forest.pkl'}")

    return model


def load():
    return joblib.load(MODELS_PATH / "random_forest.pkl")


def predict(X):
    model = load()
    return {
        "prediction": model.predict(X).tolist(),
        "proba_churn": model.predict_proba(X)[:, 1].tolist(),
    }


if __name__ == "__main__":
    train()
