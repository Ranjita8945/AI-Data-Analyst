import pandas as pd


def get_data_quality_report(df):
    """Analyze missing values, duplicates, and data types."""

    report = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Missing Percentage": (
            df.isnull().mean() * 100
        ).round(2),
        "Unique Values": df.nunique()
    })

    return report


def get_duplicate_count(df):
    """Count duplicate rows."""

    return int(df.duplicated().sum())


def get_column_types(df):
    """Separate numerical and categorical columns."""

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numerical_columns, categorical_columns


def get_statistical_summary(df):
    """Generate statistical summaries."""

    return df.describe(include="all").transpose()


def get_correlation_matrix(df):
    """Calculate correlations between numerical columns."""

    numerical_df = df.select_dtypes(
        include="number"
    )

    if numerical_df.shape[1] < 2:
        return None

    return numerical_df.corr()


def generate_basic_insights(df):
    """Generate simple, data-driven observations."""

    insights = []

    # Dataset size
    insights.append(
        f"The dataset contains {df.shape[0]} rows "
        f"and {df.shape[1]} columns."
    )

    # Missing values
    missing_count = int(df.isnull().sum().sum())

    if missing_count > 0:
        columns_with_missing = (
            df.columns[df.isnull().any()].tolist()
        )

        insights.append(
            f"There are {missing_count} missing values "
            f"across {len(columns_with_missing)} columns: "
            f"{', '.join(columns_with_missing)}."
        )
    else:
        insights.append(
            "The dataset contains no missing values."
        )

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    insights.append(
        f"The dataset contains {duplicate_count} "
        "duplicate rows."
    )

    # Numerical and categorical columns
    numerical_columns, categorical_columns = (
        get_column_types(df)
    )

    insights.append(
        f"There are {len(numerical_columns)} numerical "
        f"columns and {len(categorical_columns)} "
        "categorical columns."
    )

    # Detect columns with only one unique value
    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) == 1
    ]

    if constant_columns:
        insights.append(
            "Columns containing only one distinct value: "
            + ", ".join(constant_columns)
            + "."
        )

    return insights

