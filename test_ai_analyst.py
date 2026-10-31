import pandas as pd
from src.ai_analyst import ask_your_data

df = pd.DataFrame({
    "Name": ["Priya", "Amit", "Rahul", "Sneha", "Kiran"],
    "Age": [23, 24, 22, 25, 26],
    "Salary": [30000, 28000, 35000, 40000, 32000],
})

question = "Show the 3 employees with the highest salaries"

try:
    sql, result = ask_your_data(df, question)

    print("\nGenerated SQL:")
    print(sql)

    print("\nQuery Results:")
    print(result)

except Exception as error:
    print(f"Error: {error}")