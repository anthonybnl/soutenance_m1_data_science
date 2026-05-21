from pathlib import Path
import pandas as pd

BASE_PATH = Path(__file__).parents[1]
DATA_PATH = BASE_PATH / "data"

CSV_PATH = DATA_PATH / "customer_churn_business_dataset.csv"


def charger_donnees():
    df = pd.read_csv(CSV_PATH)

    df2 = df.drop(columns=["customer_id"])
    return df
