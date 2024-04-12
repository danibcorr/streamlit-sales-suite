# %% Librerias

import streamlit as st
import pandas as pd
from functions.data_analysis import data_pipeline

# %% Definiciones para streamlit

st.set_page_config(
    page_title = "Análisis de datos",
    page_icon = "📊"
)

st.title("📊 Análisis de datos")

# %% Main

if __name__ == '__main__':

    # Directorio del fichero para la obtencio de los datos
    path = "datasets/Datos Ventas.csv"

    # Leemos el fichero CSV con los datos
    df = pd.read_csv(path)

    # Convertimos la columna 'Fecha de venta' a formato datetime
    df['Fecha de venta'] = pd.to_datetime(df['Fecha de venta'], dayfirst=True)

    # Obtenemos los años disponibles
    list_aval_years = df["Fecha de venta"].dt.year.unique()
    
    # Selectbox te permite realizar una seleccion entre varias posibilidades
    # Ahora queremos hacer la seleccion segun los años disponibles del dataset
    year_selected = st.selectbox('Elige un año para analizar los datos', list_aval_years)
    
    # Pipeline para la obtención y visualización de los datos
    data_pipeline(df, path, year_selected)