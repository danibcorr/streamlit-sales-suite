# %% Libraries

import streamlit as st
import pandas as pd
from functions.data_analysis import data_pipeline
from functions.language_state import StateManager

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
def load_data(path):

    # We read the CSV file with the data
    df = pd.read_csv(path)

    # We convert the 'Fecha de venta' ('Date of sale') column to datetime format
    df['Fecha de venta'] = pd.to_datetime(df['Fecha de venta'], dayfirst = True)

    # Get the available years
    list_aval_years = df["Fecha de venta"].dt.year.unique()

    return df, list_aval_years

# %% Main

if __name__ == '__main__':

    # Directory of the file for data retrieval
    path = "../datasets/Datos Ventas.csv"

    # We load the data according to the indicated path
    df, list_aval_years = load_data(path)

    # Selectbox allows you to make a selection from among several possibilities
    # Now we want to make the selection according to the available years of the dataset
    language = state_manager.get_language()

    if (language == 'English') or (language == 'Inglés'):
    
        year_label = 'Select a year to analyze the data'
    
    elif (language == 'Spanish') or (language == 'Español'):
    
        year_label = 'Elige un año para analizar los datos'

    year_selected = st.selectbox(year_label, list_aval_years)
    
    # Pipeline for data retrieval and visualization
    data_pipeline(df, path, year_selected)