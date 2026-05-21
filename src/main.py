import sys
from pathlib import Path

# permet de faire les imports depuis le dossier src

BASE_DIR = Path(__file__).resolve().parents[1]
print(BASE_DIR)

if not BASE_DIR in sys.path:
    sys.path.append(str(BASE_DIR))

from src.chargement_donnees import charger_donnees
from src.traitement_null_outliers import traiter_donnees
from src.preprocessing import train_preprocess

if __name__ == "__main__":
    df = charger_donnees()
    print(df.head())

    df_data_ok = traiter_donnees(df)
    print(df_data_ok.head())

    train_preprocess(df_data_ok)
