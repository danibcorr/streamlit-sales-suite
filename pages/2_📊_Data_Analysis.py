# %% Libraries

import streamlit as st
import pandas as pd
from data.data_analysis import data_pipeline, compare_years
from web_functions.language_state import StateManager
from deep_translator import GoogleTranslator

# %% Parameters for Streamlit

translator = GoogleTranslator(source = 'en', target = 'es')
language_manager = StateManager()
language = language_manager.language

page_title = "Data analysis"
page_title = translator.translate(page_title) if language != 'English' else page_title
page_icon = "📊"

st.set_page_config(page_title = page_title, page_icon = page_icon, layout = "wide")
st.title(f"{page_icon} {page_title}")

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

    year_label = 'Select a year to analyze the data'
    year_label = year_label if language == 'English' else translator.translate(year_label)
    compare_label = 'Compare years'
    compare_label = compare_label if language == 'English' else translator.translate(compare_label)

    year_selected = st.sidebar.selectbox(year_label, list_aval_years)
    compare_years_checkbox = st.sidebar.checkbox(compare_label)

    if compare_years_checkbox:

        year1_label = 'Select the first year'
        year1_label = year1_label if language == 'English' else translator.translate(year1_label)
        year2_label = 'Select the second year'
        year2_label = year2_label if language == 'English' else translator.translate(year2_label)

        year1 = st.sidebar.selectbox(year1_label, list_aval_years)
        year2 = st.sidebar.selectbox(year2_label, list_aval_years)

        compare_years(df, year1, year2)

    else:

        data_pipeline(df, dataset_path, year_selected)

# %% Main

if __name__ == '__main__':
    
    dataset_path = "./datasets/Datos Ventas.csv"
    main(dataset_path)