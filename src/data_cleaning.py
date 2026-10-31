
import pandas as pd


def remove_duplicates(df):
    cleaned_df = df.drop_duplicates().copy()
    return cleaned_df


def fill_missing_values(
    df,
    numerical_strategy="median",
    categorical_strategy="mode"
):
    cleaned_df = df.copy()

    numerical_columns = cleaned_df.select_dtypes(
        include="number"
    ).columns

    for column in numerical_columns:
        if cleaned_df[column].isnull().any():

            if numerical_strategy == "mean":
                fill_value = cleaned_df[column].mean()

            elif numerical_strategy == "median":
                fill_value = cleaned_df[column].median()

            else:
                raise ValueError(
                    "numerical_strategy must be 'mean' or 'median'"
                )

            if pd.notna(fill_value):
                cleaned_df[column] = cleaned_df[column].fillna(
                    fill_value
                )

    categorical_columns = cleaned_df.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for column in categorical_columns:
        if cleaned_df[column].isnull().any():

            if categorical_strategy == "mode":
                mode_values = cleaned_df[column].mode(dropna=True)

                if not mode_values.empty:
                    fill_value = mode_values.iloc[0]
                else:
                    fill_value = "Unknown"

            elif categorical_strategy == "unknown":
                fill_value = "Unknown"

            else:
                raise ValueError(
                    "categorical_strategy must be 'mode' or 'unknown'"
                )

            if isinstance(cleaned_df[column].dtype, pd.CategoricalDtype):
                if fill_value not in cleaned_df[column].cat.categories:
                    cleaned_df[column] = cleaned_df[column].cat.add_categories(
                        [fill_value]
                    )

            cleaned_df[column] = cleaned_df[column].fillna(fill_value)

    return cleaned_df


def get_cleaning_summary(original_df, cleaned_df):
    original_missing = int(original_df.isnull().sum().sum())
    cleaned_missing = int(cleaned_df.isnull().sum().sum())

    original_duplicates = int(original_df.duplicated().sum())
    cleaned_duplicates = int(cleaned_df.duplicated().sum())

    summary = pd.DataFrame({
        "Metric": [
            "Original Rows",
            "Cleaned Rows",
            "Original Columns",
            "Cleaned Columns",
            "Duplicate Rows Before",
            "Duplicate Rows After",
            "Missing Values Before",
            "Missing Values After",
            "Rows Removed",
            "Missing Values Filled"
        ],
        "Value": [
            len(original_df),
            len(cleaned_df),
            len(original_df.columns),
            len(cleaned_df.columns),
            original_duplicates,
            cleaned_duplicates,
            original_missing,
            cleaned_missing,
            len(original_df) - len(cleaned_df),
            original_missing - cleaned_missing
        ]
    })

    return summary

