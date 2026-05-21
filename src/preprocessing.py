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


def train_test_split(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = sklearn_train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=17,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def train_preprocess(df: pd.DataFrame):
    X = df[variables_numeriques + variables_categorielles]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")

    # pour les variables catégorielles, on va faire du label encoding pour les variables binaires et du one hot encoding pour les autres

    binary_cols = ["gender", "discount_applied", "price_increase_last_3m"]
    one_hot_cols = [col for col in variables_categorielles if col not in binary_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("bin", OrdinalEncoder(), binary_cols),
            (
                "nom",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                one_hot_cols,
            ),
            ("num", StandardScaler(), variables_numeriques),
        ]
    )

    preprocessor.fit(X_train)
    X_train_preprocessed = preprocessor.transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_preprocessed, columns=feature_names)

    # Sauvegarde
    DATA_PROC_PATH = DATA_PATH / "processed"
    DATA_PROC_PATH.mkdir(parents=True, exist_ok=True)

    X_train_df.to_csv(DATA_PROC_PATH / "X_train.csv", index=False)
    X_test_df.to_csv(DATA_PROC_PATH / "X_test.csv", index=False)
    y_train.to_csv(DATA_PROC_PATH / "y_train.csv", index=False)
    y_test.to_csv(DATA_PROC_PATH / "y_test.csv", index=False)

    return (
        X_train_df,
        X_test_df,
        y_train,
        y_test,
        preprocessor,
    )
