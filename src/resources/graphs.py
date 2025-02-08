import streamlit as st
import pandas as pd

from src.utils import config_streamlit_page, check_credentials


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

    # Convert to DataFrame and include month names
    df_money_month = pd.DataFrame({"Mes": months_names, "Dinero": df_money_month.values})

    # Ensure correct order with pd.Categorical
    df_money_month["Mes"] = pd.Categorical(
        df_money_month["Mes"], categories=months_names, ordered=True
    )

    # Show graph in Streamlit
    st.subheader(f"Money earned per month in {year}")
    st.bar_chart(df_money_month.set_index("Mes"))


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

        # TODO: Number of products sold month/year

        # Money earned per month/year
        money_month(df=st.session_state.dataframe, year=int(selected_year))

        # TODO: Number of products sold category/year
        # TODO: Matrix confusion relation between product status and gender
        # TODO: Matrix confusion relation between product state and country
        # TODO: Number of people interested by category in a year


# First call to the config page function
config_streamlit_page(page_name="🏠 Graphs")

# Check if the credentials are available and display all the graphs
display_all_graphs(credentials_status=check_credentials())
