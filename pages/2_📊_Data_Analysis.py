# %% Libraries

import streamlit as st
import pandas as pd
from web_functions.language_state import StateManager
from data.data_analysis import data_pipeline, compare_years

# %% Definitions for streamlit

if 'language' not in st.session_state:
    
    st.session_state.language = 'English'

state_manager = StateManager(language=st.session_state.language)

language = state_manager.get_language()

if (language == 'English') or (language == 'Inglés'):

    st.set_page_config(page_title = "Data analysis", page_icon = "📊", layout="wide")
    st.title("📊 Data analysis")

elif (language == 'Spanish') or (language == 'Español'):
    
    st.set_page_config(page_title = "Análisis de datos", page_icon = "📊", layout="wide")
    st.title("📊 Análisis de datos")

# %% Functions

@st.cache_data
def load_data(path: str) -> tuple:

    """
    Load data from a CSV file and convert the 'Fecha de venta' column to datetime format.

    Args:
        path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the loaded DataFrame and a list of available years.
    """

    df = pd.read_csv(path)
    df['Fecha de venta'] = pd.to_datetime(df['Fecha de venta'], dayfirst = True)
    list_aval_years = df["Fecha de venta"].dt.year.unique()

    return df, list_aval_years

def main(dataset_path: str) -> None:

    """
    Main function to load data and perform data analysis.

    Args:
        dataset_path (str): The path to the dataset file.
    """

    df, list_aval_years = load_data(dataset_path)
    language = state_manager.get_language()

    if (language == 'English') or (language == 'Inglés'):

        year_label = 'Select a year to analyze the data'
        compare_label = 'Compare years'

    elif (language == 'Spanish') or (language == 'Español'):

        year_label = 'Elige un año para analizar los datos'
        compare_label = 'Comparar años'

    year_selected = st.sidebar.selectbox(year_label, list_aval_years)
    compare_years_checkbox = st.sidebar.checkbox(compare_label)

    if compare_years_checkbox:

        if (language == 'English') or (language == 'Inglés'):

            year1_label = 'Select the first year'
            year2_label = 'Select the second year'

        elif (language == 'Spanish') or (language == 'Español'):

            year1_label = 'Selecciona el primer año'
            year2_label = 'Selecciona el segundo año'

        year1 = st.sidebar.selectbox(year1_label, list_aval_years)
        year2 = st.sidebar.selectbox(year2_label, list_aval_years)

        compare_years(df, year1, year2)

    else:

        data_pipeline(df, dataset_path, year_selected)

# %% Main

if __name__ == '__main__':
    
    dataset_path = "./datasets/Datos Ventas.csv"
    main(dataset_path)