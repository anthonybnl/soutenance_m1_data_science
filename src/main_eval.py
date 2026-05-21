import argparse
import sys
from pathlib import Path
import pandas as pd

import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
if BASE_DIR not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.chargement_donnees import charger_donnees
from src.traitement_null_outliers import traiter_donnees
from src.preprocessing import train_preprocess


def main():
    df = charger_donnees()
    df = traiter_donnees(df)

    # chargement preprocessor
    preprocessor = joblib.load(BASE_DIR / "models" / "preprocessor.pkl")

    X = preprocessor.transform(df)

    X_all = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())

    print(X_all.head())


if __name__ == "__main__":
    main()
