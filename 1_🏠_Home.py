# %% Libraries

import streamlit as st
from web_functions.language_state import StateManager

# %% Globals

if 'language' not in st.session_state:

    st.session_state.language = 'English'

# Create a single StateManager instance and store it in session state
if'state_manager' not in st.session_state:
    
    st.session_state.state_manager = StateManager(language=st.session_state.language)

state_manager = st.session_state.state_manager

# %% Parameters

st.set_page_config(
    page_title = "Home",
    page_icon = "🏠"
)

st.title("🏠 Home")

# %% Functions

def select_language(state_manager):

    language = state_manager.get_language()

    if (language == 'English') or (language == 'Inglés'):
    
        language_options = ['English', 'Spanish']
        language = st.sidebar.selectbox('Select a language:', language_options)
    
    elif (language == 'Spanish') or (language == 'Español'):
    
        language_options = ['Inglés', 'Español']
        language = st.sidebar.selectbox('Selecciona un idioma:', language_options)
    
    state_manager.set_language(language)
    st.session_state.language = language  
    
    return language

def text_home(state_manager):

    language = state_manager.get_language()
    
    if (language == 'English') or (language == 'Inglés'):
    
        st.markdown("""
            # Abstract

            This project illustrates the application of **data analytics**, **data science** and **artificial intelligence** to extract meaningful information from sales data in second-hand markets. 
            sales data in second hand markets. It seeks to demonstrate competencies in these areas and to provide a practical solution for sales data management 
            for non-technical individuals. Several technologies are employed, which will be detailed in the subsequent sections.

            # Data analysis

            The first functionality is the “**Data analysis**” page, where users can interact with graphs generated from a CSV file collected from sales data in second-tier markets. 
            data from sales data in second-hand markets. Currently, data collection is done manually and stored in Google Drive. The goal is to migrate 
            this database to a **SQL** database, which will allow real-time access and updates from this web application.

            The graphs provide information on **sales metrics**, such as monthly profit, best-selling products, and correlations between product characteristics, including gender, condition, and product 
            including gender, condition and country of sale.
        """)

        st.image('images/item_analysis.png')

        st.markdown("""
            # Item classifier

            The second functionality is the “**Item classifier**”, which allows users to upload an image of an item for classification by an **artificial intelligence** model.
            This process seeks to automate data labeling, minimizing human intervention.

            The model, trained with **TensorFlow** and **Keras**, can identify and classify various types of items. These items can be visualized on the **Data analysis** page.

            In particular, ConvNext was used with all layers of the model frozen during training due to computational limitations (laptop with 
            RTX 3060 with 6GB RAM, i7-11800H processor with 8 cores and 16 threads, and 16GB RAM). In order to reduce memory costs, training time and inference time, we used 
            inference time, mixed-precision systems using 16-bit floating-point precision were employed. The image data, ranging from 0 to 255, were stored as 8-bit unsigned integers, 
            reducing the processing time from 5 minutes to less than 1 minute for approximately 9,000 images. To improve the learning process learning process and model convergence, 
            gradient centralization and adaptive gradient clipping were used, along with the Adam optimization algorithm with weight decay.

            Model evaluation and artifact generation were performed using MLflow.
        """)

        st.image('images/item_classifier.png')

    elif (language == 'Spanish') or (language == 'Español'):
    
        st.markdown("""
            # Resumen

            Este proyecto ilustra la aplicación de **análisis de datos**, **ciencia de datos** e **inteligencia artificial** para extraer información significativa de los 
            datos de ventas en mercados de segunda mano. Se busca demostrar competencias en estas áreas y ofrecer una solución práctica para la gestión de datos de ventas 
            para individuos no técnicos. Se emplean diversas tecnologías, que se detallarán en las secciones subsiguientes.

            # Análisis de datos

            La primera funcionalidad es la página de "**Análisis de datos**", donde los usuarios pueden interactuar con gráficos generados a partir de un archivo CSV recopilado 
            de datos de ventas en mercados de segunda mano. Actualmente, la recopilación de datos se realiza manualmente y se almacena en Google Drive. El objetivo es migrar 
            esta base de datos a una base de datos **SQL**, lo que permitirá el acceso y las actualizaciones en tiempo real desde esta aplicación web.

            Los gráficos ofrecen información sobre **métricas de ventas**, como el beneficio mensual, los productos más vendidos y las correlaciones entre las características 
            del producto, incluyendo género, condición y país de venta.
        """)

        st.image('images/item_analysis.png')

        st.markdown("""
            # Clasificador de artículos

            La segunda funcionalidad es el "**Clasificador de artículos**", que permite a los usuarios cargar una imagen de un artículo para su clasificación por un modelo 
            de **inteligencia artificial**. Este proceso busca automatizar el etiquetado de datos, minimizando la intervención humana.

            El modelo, entrenado con **TensorFlow** y **Keras**, puede identificar y clasificar varios tipos de artículos. Estos artículos pueden visualizarse en la página de **Análisis de datos**.

            En particular, se utilizó ConvNext con todas las capas del modelo congeladas durante el entrenamiento debido a limitaciones computacionales (portátil con 
            RTX 3060 con 6GB de RAM, procesador i7-11800H con 8 núcleos y 16 hilos, y 16GB de RAM). Para reducir los costos de memoria, el tiempo de entrenamiento y el 
            tiempo de inferencia, se emplearon sistemas de precisión mixta utilizando precisión de punto flotante de 16 bits. Los datos de imagen, que varían de 0 a 255, se almacenaron 
            como enteros sin signo de 8 bits, reduciendo el tiempo de procesamiento de 5 minutos a menos de 1 minuto para aproximadamente 9,000 imágenes. Para mejorar el proceso de 
            aprendizaje y la convergencia del modelo, se utilizaron la centralización de gradientes y el recorte de gradientes adaptativo, junto con el algoritmo de optimización Adam con decaimiento de pesos.

            La evaluación del modelo y la generación de artefactos se realizaron utilizando MLflow.
        """)

        st.image('images/item_classifier.png')

def main(state_manager: st.session_state.state_manager) -> None:

    select_language(state_manager)
    text_home(state_manager)

# %% Main

if __name__ == '__main__':

    main(state_manager)