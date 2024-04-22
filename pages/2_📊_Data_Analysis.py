# %% Libraries

import streamlit as st
import pandas as pd
from web_functions.language_state import StateManager
from data.data_analysis import data_pipeline

# %% Definitions for streamlit

if 'language' not in st.session_state:
    
    st.session_state.language = 'English'

state_manager = StateManager(language=st.session_state.language)

language = state_manager.get_language()

if (language == 'English') or (language == 'Inglés'):

    st.set_page_config(page_title = "Data Analysis", page_icon = "📊")
    st.title("📊 Data Analysis")

elif (language == 'Spanish') or (language == 'Español'):
    
    st.set_page_config(page_title = "Análisis de datos", page_icon = "📊")
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

    elif (language == 'Spanish') or (language == 'Español'):

        year_label = 'Elige un año para analizar los datos'

    year_selected = st.selectbox(year_label, list_aval_years)
    data_pipeline(df, dataset_path, year_selected)

# %% Main

if __name__ == '__main__':
    
    dataset_path = "./datasets/Datos Ventas.csv"
    main(dataset_path)