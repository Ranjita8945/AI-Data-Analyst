import hashlib

import pandas as pd
import streamlit as st
import plotly.express as px

from src.data_analytics import (
    get_data_quality_report,
    get_duplicate_count,
    get_column_types,
    get_statistical_summary,
    get_correlation_matrix,
    generate_basic_insights,
)

from src.visualization import (
    create_bar_chart,
    create_histogram,
    create_scatter_plot,
    create_box_plot,
)

from src.data_cleaning import (
    remove_duplicates,
    fill_missing_values,
    get_cleaning_summary,
)


from src.advanced_eda import (
    get_skewness,
    detect_outliers,
    get_categorical_summary,
    get_missing_value_summary
)

from src.sql_analytics import run_sql_query



# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# 2. APPLICATION HEADER
# =========================================================

st.title("📊 AI Data Analyst")

st.markdown(
    """
    ### Your Personal Data Analysis Assistant

    Upload a dataset to explore its structure, identify data
    quality issues, discover insights, visualize relationships,
    and automatically clean your data.
    """
)

st.sidebar.title("📊 AI Data Analyst")

st.sidebar.markdown(
    """
    **Available Features**

    - Dataset exploration
    - Data quality analysis
    - Automated insights
    - Correlation analysis
    - Interactive visualizations
    - Automated data cleaning
    - Advanced EDA
    - SQL analytics
    """
)


# =========================================================
# 3. FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"],
)


if uploaded_file is not None:

    try:

        # =================================================
        # 4. LOAD DATASET
        # =================================================

        file_bytes = uploaded_file.getvalue()

        # Identify the uploaded dataset.
        # This helps prevent using cleaning results from
        # a previously uploaded file.

        dataset_signature = hashlib.md5(
            file_bytes
        ).hexdigest()

        if st.session_state.get("dataset_signature") != dataset_signature:

            st.session_state["dataset_signature"] = dataset_signature

            st.session_state.pop("cleaned_df", None)

        if uploaded_file.name.lower().endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            df = pd.read_excel(uploaded_file)

        # Check whether the dataset contains any data.

        if df.empty or len(df.columns) == 0:

            st.warning(
                "The uploaded dataset is empty. "
                "Please upload a dataset containing data."
            )

            st.stop()

        st.success("Dataset uploaded successfully!")

        # =================================================
        # 5. DATASET OVERVIEW
        # =================================================

        st.subheader("📌 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Rows",
                f"{df.shape[0]:,}",
            )

        with col2:

            st.metric(
                "Total Columns",
                f"{df.shape[1]:,}",
            )

        with col3:

            st.metric(
                "Missing Values",
                f"{int(df.isnull().sum().sum()):,}",
            )

        with col4:

            st.metric(
                "Duplicate Rows",
                f"{get_duplicate_count(df):,}",
            )

        # =================================================
        # 6. COLUMN CLASSIFICATION
        # =================================================

        numerical_columns, categorical_columns = get_column_types(df)

        all_columns = df.columns.tolist()

        # =================================================
        # 7. CREATE NAVIGATION TABS
        # =================================================

        (
            overview_tab,
            quality_tab,
            insights_tab,
            correlation_tab,
            charts_tab,
            cleaning_tab,
            eda_tab,
            sql_tab
        ) = st.tabs(
            [
                "📄 Data Preview",
                "🧹 Data Quality",
                "💡 Automated Insights",
                "🔗 Correlation Analysis",
                "📊 Chart Explorer",
                "🧼 Data Cleaning",
                "🔬 Advanced EDA",
                "🗄️ SQL Analytics"
            ]
        )

        # =================================================
        # TAB 1: DATA PREVIEW
        # =================================================

        with overview_tab:

            st.subheader("Dataset Preview")

            number_of_rows = st.slider(
                "Number of rows to display",
                min_value=1,
                max_value=min(100, len(df)),
                value=min(10, len(df)),
                key="preview_rows",
            )

            st.dataframe(
                df.head(number_of_rows),
                use_container_width=True,
            )

            st.subheader("Statistical Summary")

            summary = get_statistical_summary(df)

            st.dataframe(
                summary,
                use_container_width=True,
            )

            st.subheader("Column Names")

            st.write(all_columns)

            st.subheader("Download Original Dataset")

            original_csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Original CSV",
                data=original_csv,
                file_name="original_dataset.csv",
                mime="text/csv",
                key="download_original",
            )

        # =================================================
        # TAB 2: DATA QUALITY
        # =================================================

        with quality_tab:

            st.subheader("🧹 Data Quality Report")

            quality_report = get_data_quality_report(df)

            st.dataframe(
                quality_report,
                use_container_width=True,
            )

            st.divider()

            st.subheader("Duplicate Records")

            duplicate_count = get_duplicate_count(df)

            if duplicate_count > 0:

                st.warning(
                    f"Found {duplicate_count} duplicate rows."
                )

            else:

                st.success(
                    "No duplicate rows were detected."
                )

            st.subheader("Numerical Columns")

            if numerical_columns:

                st.write(numerical_columns)

            else:

                st.info(
                    "No numerical columns were detected."
                )

            st.subheader("Categorical Columns")

            if categorical_columns:

                st.write(categorical_columns)

            else:

                st.info(
                    "No categorical columns were detected."
                )

            st.subheader("Missing Values by Column")

            missing_values = df.isnull().sum()

            missing_values = missing_values[
                missing_values > 0
            ]

            if not missing_values.empty:

                st.bar_chart(missing_values)

                st.dataframe(
                    missing_values.rename(
                        "Missing Values"
                    ).to_frame(),
                    use_container_width=True,
                )

            else:

                st.success(
                    "There are no missing values in this dataset."
                )

        # =================================================
        # TAB 3: AUTOMATED INSIGHTS
        # =================================================

        with insights_tab:

            st.subheader("💡 Automated Dataset Insights")

            insights = generate_basic_insights(df)

            for index, insight in enumerate(
                insights,
                start=1,
            ):

                st.markdown(
                    f"**{index}.** {insight}"
                )

            st.divider()

            st.subheader("Numerical Column Insights")

            if numerical_columns:

                selected_column = st.selectbox(
                    "Select a numerical column",
                    numerical_columns,
                    key="insight_column",
                )

                column_data = pd.to_numeric(
                    df[selected_column],
                    errors="coerce",
                ).dropna()

                if not column_data.empty:

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Mean",
                            f"{column_data.mean():,.2f}",
                        )

                    with col2:

                        st.metric(
                            "Median",
                            f"{column_data.median():,.2f}",
                        )

                    with col3:

                        standard_deviation = (
                            column_data.std()
                            if len(column_data) > 1
                            else None
                        )

                        st.metric(
                            "Standard Deviation",
                            (
                                f"{standard_deviation:,.2f}"
                                if standard_deviation is not None
                                else "N/A"
                            ),
                        )

                    st.write(
                        f"Minimum: {column_data.min():,.2f}"
                    )

                    st.write(
                        f"Maximum: {column_data.max():,.2f}"
                    )

                else:

                    st.info(
                        "No valid numerical values are available."
                    )

            else:

                st.info(
                    "This dataset has no numerical columns."
                )

        # =================================================
        # TAB 4: CORRELATION ANALYSIS
        # =================================================

        with correlation_tab:

            st.subheader("🔗 Correlation Analysis")

            correlation_matrix = get_correlation_matrix(df)

            if correlation_matrix is not None:

                st.write(
                    """
                    Correlation measures the strength and direction
                    of a linear relationship between numerical
                    variables.

                    Values range from -1 to +1.
                    """
                )

                st.dataframe(
                    correlation_matrix.style.format("{:.2f}"),
                    use_container_width=True,
                )

                # -----------------------------------------
                # CORRELATION HEATMAP
                # -----------------------------------------

                st.subheader("Correlation Heatmap")

                fig = px.imshow(
                    correlation_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Correlation Heatmap",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                # -----------------------------------------
                # STRONGEST CORRELATIONS
                # -----------------------------------------

                st.subheader("Strongest Correlations")

                if correlation_matrix.shape[1] >= 2:

                    upper_triangle_mask = pd.DataFrame(
                        [
                            [
                                i < j
                                for j in range(
                                    len(correlation_matrix.columns)
                                )
                            ]
                            for i in range(
                                len(correlation_matrix.columns)
                            )
                        ],
                        index=correlation_matrix.index,
                        columns=correlation_matrix.columns,
                    )

                    upper_triangle = correlation_matrix.where(
                        upper_triangle_mask
                    )

                    strongest_correlations = (
                        upper_triangle.stack()
                        .rename("Correlation")
                        .reset_index()
                    )

                    strongest_correlations.columns = [
                        "Variable 1",
                        "Variable 2",
                        "Correlation",
                    ]

                    strongest_correlations[
                        "Absolute Correlation"
                    ] = strongest_correlations[
                        "Correlation"
                    ].abs()

                    strongest_correlations = (
                        strongest_correlations.sort_values(
                            "Absolute Correlation",
                            ascending=False,
                        )
                        .head(10)
                        .drop(
                            columns=["Absolute Correlation"]
                        )
                    )

                    if not strongest_correlations.empty:

                        st.dataframe(
                            strongest_correlations,
                            use_container_width=True,
                        )

                    else:

                        st.info(
                            "No variable pairs are available."
                        )

            else:

                st.info(
                    "At least two numerical columns are required "
                    "for correlation analysis."
                )

        # =================================================
        # TAB 5: INTERACTIVE CHART EXPLORER
        # =================================================

        with charts_tab:

            st.subheader("📊 Interactive Chart Explorer")

            st.write(
                "Choose a chart type and select the columns "
                "you want to visualize."
            )

            chart_type = st.selectbox(
                "Choose Chart Type",
                [
                    "Bar Chart",
                    "Histogram",
                    "Scatter Plot",
                    "Box Plot",
                ],
                key="chart_type",
            )

            # ---------------------------------------------
            # BAR CHART
            # ---------------------------------------------

            if chart_type == "Bar Chart":

                st.markdown("#### Bar Chart")

                x_column = st.selectbox(
                    "Select a column for the X-axis",
                    all_columns,
                    key="bar_x",
                )

                use_average = st.checkbox(
                    "Calculate the average of a numerical column",
                    key="bar_average",
                )

                if use_average:

                    if numerical_columns:

                        y_column = st.selectbox(
                            "Select a numerical column",
                            numerical_columns,
                            key="bar_y",
                        )

                        fig = create_bar_chart(
                            df,
                            x_column,
                            y_column,
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                    else:

                        st.warning(
                            "No numerical columns are available."
                        )

                else:

                    fig = create_bar_chart(
                        df,
                        x_column,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

            # ---------------------------------------------
            # HISTOGRAM
            # ---------------------------------------------

            elif chart_type == "Histogram":

                st.markdown("#### Histogram")

                if numerical_columns:

                    column = st.selectbox(
                        "Select a numerical column",
                        numerical_columns,
                        key="hist_column",
                    )

                    fig = create_histogram(
                        df,
                        column,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                else:

                    st.warning(
                        "A histogram requires a numerical column."
                    )

            # ---------------------------------------------
            # SCATTER PLOT
            # ---------------------------------------------

            elif chart_type == "Scatter Plot":

                st.markdown("#### Scatter Plot")

                if len(numerical_columns) >= 2:

                    col1, col2 = st.columns(2)

                    with col1:

                        x_column = st.selectbox(
                            "Select X-axis",
                            numerical_columns,
                            key="scatter_x",
                        )

                    with col2:

                        y_options = [
                            column
                            for column in numerical_columns
                            if column != x_column
                        ]

                        y_column = st.selectbox(
                            "Select Y-axis",
                            y_options,
                            key="scatter_y",
                        )

                    fig = create_scatter_plot(
                        df,
                        x_column,
                        y_column,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                else:

                    st.warning(
                        "A scatter plot requires at least "
                        "two numerical columns."
                    )

            # ---------------------------------------------
            # BOX PLOT
            # ---------------------------------------------

            elif chart_type == "Box Plot":

                st.markdown("#### Box Plot")

                if numerical_columns:

                    y_column = st.selectbox(
                        "Select a numerical column",
                        numerical_columns,
                        key="box_y",
                    )

                    group_by = st.checkbox(
                        "Group by another column",
                        key="box_group_enabled",
                    )

                    if group_by:

                        group_options = [
                            column
                            for column in all_columns
                            if column != y_column
                        ]

                        if group_options:

                            x_column = st.selectbox(
                                "Select a grouping column",
                                group_options,
                                key="box_x",
                            )

                            fig = create_box_plot(
                                df,
                                x_column,
                                y_column,
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                            )

                        else:

                            st.info(
                                "No other columns are available "
                                "for grouping."
                            )

                    else:

                        fig = create_box_plot(
                            df,
                            y_column,
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                else:

                    st.warning(
                        "A box plot requires a numerical column."
                    )

        # =================================================
        # TAB 6: AUTOMATED DATA CLEANING
        # =================================================

        with cleaning_tab:

            st.subheader("🧼 Automated Data Cleaning")

            st.write(
                """
                Clean your dataset by removing duplicate rows
                and filling missing values.

                Your original dataset will remain unchanged.
                """
            )

    
            # ---------------------------------------------
            # CLEANING OPTIONS
            # ---------------------------------------------

            st.markdown("### 1. Choose Cleaning Options")

            remove_duplicate_rows = st.checkbox(
                "Remove duplicate rows",
                value=True,
                key="clean_duplicates",
            )

            numerical_strategy = st.selectbox(
                "How should missing numerical values be filled?",
                ["median", "mean"],
                key="clean_numerical_strategy",
            )

            categorical_strategy = st.selectbox(
                "How should missing categorical values be filled?",
                ["mode", "unknown"],
                key="clean_categorical_strategy",
            )

            st.caption(
                "Median is less affected by extreme numerical values "
                "than the mean. Mode fills missing categorical values "
                "with the most frequent category."
            )




            # ---------------------------------------------
            # RUN CLEANING
            # ---------------------------------------------

            st.markdown("### 2. Clean Your Dataset")

            if st.button(
                "🧹 Clean Dataset",
                key="run_cleaning",
                type="primary",
            ):

                cleaned_df = df.copy()

                # Remove duplicates if selected.

                if remove_duplicate_rows:

                    cleaned_df = remove_duplicates(
                        cleaned_df
                    )

                # Fill missing values.

                cleaned_df = fill_missing_values(
                    cleaned_df,
                    numerical_strategy=numerical_strategy,
                    categorical_strategy=categorical_strategy,
                )

                # Save the cleaned dataset in Streamlit
                # session state so it remains available
                # when the application reruns.

                st.session_state["cleaned_df"] = cleaned_df

                st.success(
                    "Dataset cleaning completed successfully!"
                )

            # ---------------------------------------------
            # DISPLAY CLEANING RESULTS
            # ---------------------------------------------

            if "cleaned_df" in st.session_state:

                cleaned_df = st.session_state["cleaned_df"]

                st.divider()

                st.markdown("### 3. Cleaning Summary")

                cleaning_summary = get_cleaning_summary(
                    df,
                    cleaned_df,
                )

                st.dataframe(
                    cleaning_summary,
                    use_container_width=True,
                    hide_index=True,
                )

                # -----------------------------------------
                # REMAINING MISSING VALUES
                # -----------------------------------------

                remaining_missing = int(
                    cleaned_df.isnull().sum().sum()
                )

                if remaining_missing > 0:

                    st.warning(
                        f"{remaining_missing} missing values remain. "
                        "Some columns may contain only missing values "
                        "or require additional cleaning rules."
                    )

                else:

                    st.success(
                        "No missing values remain in the cleaned dataset."
                    )

                # -----------------------------------------
                # CLEANED DATA PREVIEW
                # -----------------------------------------

                st.markdown("### 4. Cleaned Dataset Preview")

                st.dataframe(
                    cleaned_df.head(20),
                    use_container_width=True,
                )

                st.write(
                    f"**Rows:** {cleaned_df.shape[0]:,} | "
                    f"**Columns:** {cleaned_df.shape[1]:,}"
                )

                # -----------------------------------------
                # DOWNLOAD CLEANED CSV
                # -----------------------------------------

                st.markdown("### 5. Download Cleaned Dataset")

                cleaned_csv = cleaned_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Cleaned CSV",
                    data=cleaned_csv,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv",
                    key="download_cleaned_csv",
                )



        # =================================================
        # TAB 7: ADVANCED EDA
        # =================================================

        with eda_tab:
            st.subheader("Advanced Exploratory Data Analysis")

            st.write(
                "Explore data distributions, outliers, skewness, "
                "categorical values, and missing values."
            )

            st.markdown("### 1. Skewness Analysis")

            skewness_df = get_skewness(df)

            if not skewness_df.empty:
                st.dataframe(skewness_df, use_container_width=True)
            else:
                st.info("No numerical columns available for skewness analysis.")

            st.markdown("### 2. Outlier Detection")

            outliers_df = detect_outliers(df)

            if not outliers_df.empty:
                st.dataframe(outliers_df, use_container_width=True)

                selected_column = st.selectbox(
                    "Choose a numerical column to inspect",
                    outliers_df["Column"].tolist(),
                    key="eda_outlier_column"
                )

                numerical_values = df[selected_column].dropna()

                q1 = numerical_values.quantile(0.25)
                q3 = numerical_values.quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outlier_values = numerical_values[
                    (numerical_values < lower_bound)
                    | (numerical_values > upper_bound)
                ]

                st.write(f"Detected outliers: {len(outlier_values)}")

                st.plotly_chart(
                    create_box_plot(df, selected_column),
                    use_container_width=True
                )
            else:
                st.info("No numerical columns available for outlier detection.")

            st.markdown("### 3. Categorical Analysis")

            categorical_df = get_categorical_summary(df)

            if not categorical_df.empty:
                st.dataframe(categorical_df, use_container_width=True)
            else:
                st.info("No categorical columns available.")

            st.markdown("### 4. Missing Values Analysis")

            missing_df = get_missing_value_summary(df)

            st.dataframe(missing_df, use_container_width=True)

            missing_chart_df = missing_df[
                missing_df["Missing Values"] > 0
            ]

            if not missing_chart_df.empty:
                import plotly.express as px

                fig = px.bar(
                    missing_chart_df,
                    x="Column",
                    y="Missing Values",
                    title="Missing Values by Column",
                    text="Missing Values"
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found in the dataset.")

            st.markdown("### 5. Download EDA Reports")

            report_csv = pd.concat(
                [
                    skewness_df.assign(Report="Skewness"),
                    outliers_df.assign(Report="Outliers"),
                    categorical_df.assign(Report="Categorical"),
                    missing_df.assign(Report="Missing Values")
                ],
                ignore_index=True,
                sort=False
            )

            st.download_button(
                label="Download EDA Report",
                data=report_csv.to_csv(index=False).encode("utf-8"),
                file_name="advanced_eda_report.csv",
                mime="text/csv"
            )

        
        with sql_tab:
            st.subheader("🗄️ SQL Analytics")

            st.write(
                "Write SQL queries to analyze your uploaded dataset."
            )

            st.info(
                "Use uploaded_data as the table name in your SQL queries."
            )

            st.markdown("### Dataset Columns")

            st.dataframe(
                pd.DataFrame({
                    "Column Name": df.columns,
                    "Data Type": df.dtypes.astype(str).values
                }),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Example Queries")

            query_examples = {
                "Show first 10 rows": """
SELECT *
FROM uploaded_data
LIMIT 10;
""",
                "Count total rows": """
SELECT COUNT(*) AS total_rows
FROM uploaded_data;
""",
                "Find missing values": None,
                "Top 5 rows by a numerical column": None,
                "Count records by category": None
            }

            selected_example = st.selectbox(
                "Choose an example query",
                list(query_examples.keys()),
                key="sql_example"
            )

            example_query = query_examples[selected_example]

            if selected_example == "Find missing values":
                missing_conditions = " + ".join(
                    [
                        f"COUNT(*) FILTER (WHERE \"{column}\" IS NULL)"
                        for column in df.columns
                    ]
                )

                example_query = (
                    f"SELECT {missing_conditions} AS total_missing_values "
                    "FROM uploaded_data;"
                )

            elif selected_example == "Top 5 rows by a numerical column":
                if numerical_columns:
                    selected_numeric = st.selectbox(
                        "Choose a numerical column",
                        numerical_columns,
                        key="sql_numeric_column"
                    )

                    example_query = (
                        f'SELECT * FROM uploaded_data '
                        f'ORDER BY "{selected_numeric}" DESC LIMIT 5;'
                    )
                else:
                    example_query = "SELECT * FROM uploaded_data LIMIT 5;"

            elif selected_example == "Count records by category":
                if categorical_columns:
                    selected_category = st.selectbox(
                        "Choose a categorical column",
                        categorical_columns,
                        key="sql_category_column"
                    )

                    example_query = (
                        f'SELECT "{selected_category}", COUNT(*) AS total '
                        f'FROM uploaded_data '
                        f'GROUP BY "{selected_category}" '
                        f'ORDER BY total DESC;'
                    )
                else:
                    example_query = (
                        "SELECT COUNT(*) AS total_rows FROM uploaded_data;"
                    )

            st.markdown("### SQL Query Editor")

            sql_query = st.text_area(
                "Enter your SQL query",
                value=example_query,
                height=180,
                key="sql_query_editor"
            )

            if st.button("▶️ Run SQL Query", key="run_sql_query"):
                try:
                    if not sql_query.strip():
                        st.warning("Please enter a SQL query.")
                    else:
                        normalized_query = sql_query.strip().lower()

                        if not (
                            normalized_query.startswith("select ")
                            or normalized_query.startswith("with ")
                        ):
                            st.error(
                                "Only SELECT and WITH queries are allowed."
                            )
                        else:
                            result_df = run_sql_query(df, sql_query)

                            st.session_state["sql_result_df"] = result_df

                            st.success(
                                f"Query executed successfully. "
                                f"{len(result_df):,} rows returned."
                            )

                except Exception as error:
                    st.error(f"SQL query failed: {error}")

            if "sql_result_df" in st.session_state:
                result_df = st.session_state["sql_result_df"]

                st.markdown("### Query Results")

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    label="⬇️ Download Query Results",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="sql_query_results.csv",
                    mime="text/csv",
                    key="download_sql_results"
                )


    

    # =====================================================
    # 8. ERROR HANDLING
    # =====================================================

    except Exception as error:

        st.error(
            f"Unable to analyze the dataset: {error}"
        )

        st.info(
            "Check the uploaded file, column data types, "
            "and terminal output for additional details."
        )


# =========================================================
# 9. INITIAL SCREEN
# =========================================================

else:

    st.info(
        "👆 Upload a CSV or Excel file to begin your analysis."
    )

    st.markdown(
        """
        ### What you can do with this application

        - Explore datasets.
        - Inspect missing values and duplicate records.
        - Generate automated observations.
        - Analyze correlations.
        - Create interactive visualizations.
        - Automatically clean datasets.
        - Download cleaned datasets.

        **Supported formats:** CSV and Excel (.xlsx)
        """
    )