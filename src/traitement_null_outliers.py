import pandas as pd


def traiter_donnees(df: pd.DataFrame) -> pd.DataFrame:
    df_data_ok = df.copy()
    df_data_ok["complaint_type"] = df_data_ok["complaint_type"].fillna("No_Complaint")

    variables_outliers = [
        "avg_session_time",
        "features_used",
        "usage_growth_rate",
        "last_login_days_ago",
        "monthly_fee",
        "total_revenue",
    ]

    for col in variables_outliers:
        Q1 = df_data_ok[col].quantile(0.25)
        Q3 = df_data_ok[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df_data_ok[(df_data_ok[col] < lower) | (df_data_ok[col] > upper)]
        print(
            f"Variable : {col}, remplacement de {len(outliers)} outliers par les bornes"
        )
        df_data_ok[col] = df_data_ok[col].clip(lower=lower, upper=upper)

    return df_data_ok
