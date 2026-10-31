
import pandas as pd
import plotly.express as px


def create_bar_chart(df, x_column, y_column=None):
    """Create a bar chart."""

    if y_column:
        chart_data = (
            df.groupby(x_column, dropna=False)[y_column]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            chart_data,
            x=x_column,
            y=y_column,
            title=f"Average {y_column} by {x_column}"
        )

    else:
        chart_data = (
            df[x_column]
            .fillna("Missing")
            .value_counts()
            .rename_axis(x_column)
            .reset_index(name="Count")
        )

        fig = px.bar(
            chart_data,
            x=x_column,
            y="Count",
            title=f"Count of {x_column}"
        )

    return fig


def create_histogram(df, column):
    """Create a histogram for a numerical column."""

    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}",
        marginal="box"
    )

    return fig


def create_scatter_plot(df, x_column, y_column):
    """Create a scatter plot."""

    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        title=f"{y_column} vs {x_column}",
        trendline="ols"
    )

    return fig


def create_box_plot(df, x_column, y_column=None):
    """Create a box plot."""

    if y_column:
        fig = px.box(
            df,
            x=x_column,
            y=y_column,
            title=f"Distribution of {y_column} by {x_column}"
        )
    else:
        fig = px.box(
            df,
            y=x_column,
            title=f"Distribution of {x_column}"
        )

    return fig

