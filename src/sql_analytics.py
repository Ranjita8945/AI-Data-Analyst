
import duckdb
import pandas as pd


def run_sql_query(df, query):
    connection = duckdb.connect(database=":memory:")

    try:
        connection.register("uploaded_data", df)

        result = connection.execute(query).fetchdf()

        return result

    finally:
        connection.close()


def get_sample_queries():
    return {
        "Show first 10 rows": """
SELECT *
FROM uploaded_data
LIMIT 10;
""",
        "Count total rows": """
SELECT COUNT(*) AS total_rows
FROM uploaded_data;
""",
        "Find missing values": """
SELECT
    COUNT(*) FILTER (WHERE column_name IS NULL) AS missing_values
FROM uploaded_data;
""",
        "Top 5 highest values in a numerical column": """
SELECT *
FROM uploaded_data
ORDER BY numerical_column DESC
LIMIT 5;
""",
        "Count records by category": """
SELECT column_name, COUNT(*) AS total
FROM uploaded_data
GROUP BY column_name
ORDER BY total DESC;
"""
    }

if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "Name": ["Rahul", "Priya", "Amit"],
        "Age": [22, 23, 24],
        "Salary": [25000, 30000, 28000]
    })

    result = run_sql_query(
        sample_df,
        "SELECT * FROM uploaded_data WHERE Salary > 26000"
    )

    print(result)