import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from src.chargement_donnees import charger_donnees
from src.traitement_null_outliers import traiter_donnees
from src.commun import variables_categorielles, variables_numeriques
from sklearn.model_selection import train_test_split as sklearn_train_test_split

BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_PATH / "data"
DATA_PROC_PATH = DATA_PATH / "processed"


def charger_donnees_preprocessed():
    X_train = pd.read_csv(DATA_PROC_PATH / "X_train.csv", dtype="float64")
    X_test = pd.read_csv(DATA_PROC_PATH / "X_test.csv", dtype="float64")
    y_train = pd.read_csv(DATA_PROC_PATH / "y_train.csv", dtype="float64")
    y_test = pd.read_csv(DATA_PROC_PATH / "y_test.csv", dtype="float64")

    # y_train = y_train.iloc[:,0].astype("float64")
    # y_test = y_test.iloc[:,0].astype("float64")

    y_train = y_train.iloc[:, 0]
    y_test = y_test.iloc[:, 0]

    return X_train, X_test, y_train, y_test
