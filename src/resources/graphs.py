import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

from src.utils import config_streamlit_page, check_credentials, obtain_top

# List of month names
months_names = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def money_month(df: pd.DataFrame, year: int) -> None:
    """
    Calculates and visualizes the total money earned per month for a given year.

    Args:
        df: The DataFrame containing sales data.
        year: The year for which to calculate the total money earned per month.
    """

    # First we filter by year
    df_filtered = df[df["Fecha de venta"].dt.year == year].copy()

    # Then we create a column by month
    df_filtered["Mes"] = df_filtered["Fecha de venta"].dt.month

    # We group the money obtained per month
    df_money_month = df_filtered.groupby("Mes")["Precio producto"].sum()

    # We fill in those months without data with values at 0
    all_months = range(1, 13)
    df_money_month = df_money_month.reindex(all_months, fill_value=0)

    # Convert to DataFrame and include month names
    df_money_month = pd.DataFrame({"Mes": months_names, "Dinero": df_money_month.values})

    # Ensure correct order with pd.Categorical
    df_money_month["Mes"] = pd.Categorical(
        df_money_month["Mes"], categories=months_names, ordered=True
    )

    # Show graph in Streamlit
    st.subheader(f"Money earned per month in {year}")
    st.bar_chart(df_money_month.set_index("Mes"), x_label="Month", y_label="Cash")


def product_month(df: pd.DataFrame, year: int) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        year (int): _description_
    """

    st.subheader(f"Products sold per month in {year}")

    col1, col2 = st.columns(2, gap="large")

    # Filter data by year
    df_filtered_year = df[df["Fecha de venta"].dt.year == year].copy()

    df_filtered = df_filtered_year.copy()
    df_filtered["Mes"] = df_filtered_year["Fecha de venta"].dt.month
    df_filtered = (
        df_filtered.groupby(["Mes", "Tipo producto"]).size().reset_index(name="count")
    )

    # Convert month numbers to month names
    df_filtered["Mes"] = df_filtered["Mes"].apply(
        lambda x: months_names[x - 1]
    )  # Map month numbers

    # Ensure correct order with pd.Categorical
    df_filtered["Mes"] = pd.Categorical(
        df_filtered["Mes"], categories=months_names, ordered=True
    )

    # Pivot for Streamlit bar chart
    df_pivotado = df_filtered.pivot(
        index="Mes", columns="Tipo producto", values="count"
    ).fillna(0)

    # Remove months with no products
    df_pivotado = df_pivotado.loc[(df_pivotado > 0).any(axis=1)]

    with col1:
        # Display chart in Streamlit
        st.bar_chart(df_pivotado, horizontal=True, x_label="Products", y_label="Month")
    with col2:
        top_3_products = obtain_top(df=df_filtered_year, top=3, column="Tipo producto")
        st.metric(label="Top 3 products", value=str(top_3_products), border=True)

        a, b = st.columns(2)
        a.metric(
            label="Total products sold",
            value=int(df_filtered_year["Tipo producto"].value_counts().sum()),
            border=True,
        )
        b.metric(
            label="Total cash obtained",
            value=str(float(df_filtered_year["Precio producto"].sum().round(4))) + " €",
            border=True,
        )


def gender_status(df: pd.DataFrame, year: int) -> None:
    """
    Calculate and visualize the gender distribution by product status for a given year.

    Args:
        df: The sales data.
        year: The year to filter the data for.

    Returns:
        plotly.graph_objs.Figure: The heatmap figure.
    """

    st.subheader("Gender Status Heatmap")

    # Filter data for the specified year
    data_year = df[df["Fecha de venta"].dt.year == year].copy()

    # Group by gender
    df_2dhist = pd.DataFrame(
        {
            x_label: grp["Estado del producto"].value_counts()
            for x_label, grp in data_year.groupby("Genero")
        }
    )

    # Create heatmap using Plotly with a recognized color scale
    graph_gender_status = px.imshow(
        df_2dhist, text_auto=True, color_continuous_scale="mint"
    )

    # Create heatmap analysis figures using subplots
    fig_heatmaps = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"rowspan": 2}, {}, None], [{}, None, None]],
        horizontal_spacing=0.15,
    )

    fig_heatmaps.add_trace(graph_gender_status["data"][0], row=1, col=2)

    # Adjust layout settings for better visualization
    fig_heatmaps.update_layout(
        height=800,
    )
    fig_heatmaps.update_coloraxes(showscale=False)

    # Display the figure in Streamlit
    st.plotly_chart(fig_heatmaps, use_container_width=True, theme=None)


def status_country(df: pd.DataFrame, year: int) -> None:
    """
    Calculate and visualize the product status distribution by country for a given year.

    Args:
        df: The sales data.
        year: The year to filter the data for.

    Returns:
        plotly.graph_objs.Figure: The heatmap figure.
    """

    st.subheader("Product Status by Country Heatmap")

    # Filter data for the specified year
    data_year = df[df["Fecha de venta"].dt.year == year].copy()

    # Group by product status and count the occurrences by country
    df_2dhist = pd.DataFrame(
        {
            x_label: grp["Pais"].value_counts()
            for x_label, grp in data_year.groupby("Estado del producto")
        }
    )

    # Create the heatmap using Plotly
    graph_status_country = px.imshow(
        df_2dhist, text_auto=True, color_continuous_scale="mint"
    )

    # Create a subplot for the heatmap
    fig_heatmaps = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"rowspan": 2}, {}, None], [{}, None, None]],
        horizontal_spacing=0.15,
    )

    # Add the heatmap to the subplot
    fig_heatmaps.add_trace(graph_status_country["data"][0], row=1, col=1)

    # Adjust layout settings for better visualization
    fig_heatmaps.update_layout(height=800)
    fig_heatmaps.update_coloraxes(showscale=False)

    # Display the figure in Streamlit
    st.plotly_chart(fig_heatmaps, use_container_width=True, theme=None)


def display_all_graphs(credentials_status: bool) -> None:
    """
    Displays various graphs based on the availability of credentials.

    Args:
        credentials_status: The status indicating whether the user has valid credentials.
    """

    if credentials_status:
        # Show a select box with the years available in the dataset
        available_years = st.session_state.dataframe["Fecha de venta"].dt.year.unique()
        selected_year: str = st.selectbox("Select a year", available_years)

        # Number of products sold month/year
        product_month(df=st.session_state.dataframe, year=int(selected_year))

        # Money earned per month/year
        money_month(df=st.session_state.dataframe, year=int(selected_year))

        col1, col2 = st.columns(2)
        with col1:
            # Matrix confusion relation between product status and gender
            gender_status(df=st.session_state.dataframe, year=int(selected_year))
        with col2:
            # Matrix confusion relation between product state and country
            status_country(df=st.session_state.dataframe, year=int(selected_year))

        # TODO: Number of people interested by category in a year


# First call to the config page function
config_streamlit_page(page_name="🏠 Graphs")

# Check if the credentials are available and display all the graphs
display_all_graphs(credentials_status=check_credentials())
