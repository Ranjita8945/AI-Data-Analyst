
import pandas as pd
import numpy as np


def get_skewness(df):
    numerical_df = df.select_dtypes(include="number")

    if numerical_df.empty:
        return pd.DataFrame(columns=["Column", "Skewness", "Interpretation"])

    skewness = numerical_df.skew()

    results = []

    for column, value in skewness.items():
        if pd.isna(value):
            interpretation = "Insufficient data"

        elif value > 0.5:
            interpretation = "Right-skewed"

        elif value < -0.5:
            interpretation = "Left-skewed"

        else:
            interpretation = "Approximately symmetric"

        results.append({
            "Column": column,
            "Skewness": round(value, 3) if pd.notna(value) else np.nan,
            "Interpretation": interpretation
        })

    return pd.DataFrame(results)


def detect_outliers(df):
    numerical_df = df.select_dtypes(include="number")
    results = []

    for column in numerical_df.columns:
        values = numerical_df[column].dropna()

        if values.empty:
            results.append({
                "Column": column,
                "Lower Bound": np.nan,
                "Upper Bound": np.nan,
                "Outlier Count": 0,
                "Outlier Percentage": 0.0
            })
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(
            ((values < lower_bound) | (values > upper_bound)).sum()
        )

        outlier_percentage = (outlier_count / len(values)) * 100

        results.append({
            "Column": column,
            "Lower Bound": round(lower_bound, 3),
            "Upper Bound": round(upper_bound, 3),
            "Outlier Count": outlier_count,
            "Outlier Percentage": round(outlier_percentage, 2)
        })

    return pd.DataFrame(results)


def get_categorical_summary(df):
    categorical_df = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    )

    results = []

    for column in categorical_df.columns:
        results.append({
            "Column": column,
            "Unique Categories": int(df[column].nunique()),
            "Most Frequent Value": (
                str(df[column].mode().iloc[0])
                if not df[column].mode().empty
                else "No data"
            ),
            "Most Frequent Count": int(
                df[column].value_counts().iloc[0]
            ) if not df[column].value_counts().empty else 0
        })

    return pd.DataFrame(results)


def get_missing_value_summary(df):
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df) * 100) if len(df) else missing_count.astype(float)

    summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing_count.values,
        "Missing Percentage": missing_percentage.values
    })

    return summary.sort_values(
        by="Missing Values",
        ascending=False
    ).reset_index(drop=True)

