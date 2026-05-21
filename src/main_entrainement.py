import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if BASE_DIR not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.chargement_donnees import charger_donnees
from src.traitement_null_outliers import traiter_donnees
from src.preprocessing import train_preprocess

MODELS = {
    "xgboost": "src.model_xgboost",
    "random_forest": "src.model_random_forest",
    "logistic_regression": "src.model_logistic_regression",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline complet : preprocessing + entraînement")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="xgboost",
        help="Modèle à entraîner (défaut : xgboost)",
    )
    args = parser.parse_args()

    print("=== Chargement des données ===")
    df = charger_donnees()

    print("=== Traitement des valeurs manquantes & outliers ===")
    df_data_ok = traiter_donnees(df)

    print("=== Preprocessing ===")
    train_preprocess(df_data_ok)

    print(f"=== Entraînement du modèle : {args.model} ===")
    import importlib
    module = importlib.import_module(MODELS[args.model])
    module.train()

def main_mlp():
    import torch
    
    from src.train_eval_mlp import charger_modele_mlp
    df = charger_donnees()
    print(df.head())

    df_data_ok = traiter_donnees(df)
    print(df_data_ok.head())

    X_train_df, X_test_df, y_train, y_test, preprocessor = train_preprocess(df_data_ok)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_mlp = entrainer_modele_mlp(X_train_df, y_train, X_test_df, y_test)

    # teste en chargant le modèle

    modele_sauvegarde = charger_modele_mlp(DEVICE)

    single_data = X_test_df.sample(n=1).to_numpy()

    single_data = torch.tensor(single_data, dtype=torch.float32)
    single_data = single_data.to(DEVICE)

    print(single_data.shape)

    y = modele_sauvegarde.forward(single_data)

    # le modèle sort un logit, il faut appliquer sigmoid
    proba = torch.sigmoid(y)[0].item() * 100

    print(f"Probabilité de churn : {proba:.2f}")