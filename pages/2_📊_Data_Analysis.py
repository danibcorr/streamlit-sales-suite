# %% Libraries

import streamlit as st
import pandas as pd
from data.data_analysis import data_pipeline, compare_years
from web_functions.language_state import StateManager
from deep_translator import GoogleTranslator

# %% Parameters for Streamlit

# Create a Google translator object to translate text from English to Spanish
translator = GoogleTranslator(source = 'en', target = 'es')

# Create a language state manager to manage the language state
language_manager = StateManager()
language = language_manager.language
print(language)

# Set the page title and icon
page_title = "Data analysis"
page_title = translator.translate(page_title) if language!= 'English' else page_title
page_icon = "📊"

# Set the page configuration
st.set_page_config(page_title = page_title, page_icon = page_icon, layout = "wide")

# Set the page title
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
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(path)
    
    # Convert the 'Fecha de venta' column to datetime format
    df['Fecha de venta'] = pd.to_datetime(df['Fecha de venta'], dayfirst = True)
    
    # Get a list of unique years from the 'Fecha de venta' column
    list_aval_years = df["Fecha de venta"].dt.year.unique()
    
    # Return the loaded DataFrame and the list of available years
    return df, list_aval_years

def main(dataset_path: str) -> None:

    """
    Main function to load data and perform data analysis.

    Args:
        dataset_path (str): The path to the dataset file.
    """
    
    # Load the data from the CSV file
    df, list_aval_years = load_data(dataset_path)
    
    # Set the labels for the year selection and comparison
    year_label = 'Select a year to analyze the data'
    year_label = year_label if language == 'English' else translator.translate(year_label)
    compare_label = 'Compare years'
    compare_label = compare_label if language == 'English' else translator.translate(compare_label)
    
    # Create a selectbox for year selection
    year_selected = st.sidebar.selectbox(year_label, list_aval_years)
    
    # Create a checkbox for year comparison
    compare_years_checkbox = st.sidebar.checkbox(compare_label)
    
    # If the comparison checkbox is checked, compare two years
    if compare_years_checkbox:

        # Set the labels for the first and second year selection
        year1_label = 'Select the first year'
        year1_label = year1_label if language == 'English' else translator.translate(year1_label)
        year2_label = 'Select the second year'
        year2_label = year2_label if language == 'English' else translator.translate(year2_label)
        
        # Create selectboxes for the first and second year selection
        year1 = st.sidebar.selectbox(year1_label, list_aval_years)
        year2 = st.sidebar.selectbox(year2_label, list_aval_years)
        
        # Compare the two years
        compare_years(df, year1, year2, language)
    
    # If the comparison checkbox is not checked, analyze a single year
    else:

        # Analyze the selected year
        data_pipeline(df, dataset_path, year_selected, language)

# %% Main

if __name__ == '__main__':

    dataset_path = "datasets/Datos Ventas.csv"
    main(dataset_path)