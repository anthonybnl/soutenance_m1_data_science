import pandas as pd


def traiter_donnees(df):
    df["complaint_type"] = df["complaint_type"].fillna("No_Complaint")

    variables_outliers = [
        "avg_session_time",
        "features_used",
        "usage_growth_rate",
        "last_login_days_ago",
        "monthly_fee",
        "total_revenue",
    ]

    for col in variables_outliers:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(
            f"Variable : {col}, remplacement de {len(outliers)} outliers par les bornes"
        )
        df[col] = df[col].clip(lower=lower, upper=upper)
