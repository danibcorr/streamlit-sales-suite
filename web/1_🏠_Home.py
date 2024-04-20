# %% Libraries

import streamlit as st
from functions.language_state import StateManager

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
            # Welcome

            This project is a demonstration of my skills in **data analysis**, **data science**, and the use of **artificial intelligence** techniques. I have used a variety of technologies to develop this project, including **Pandas**, **TensorFlow**, **Keras**, and more. My goal is to show how these tools can be used to extract valuable insights from data and solve complex problems.

            ## Data Analysis

            The first feature I want to present is the "**Data Analysis**" page. In this section, you will be able to interact with various graphs that I have generated from datasets. These graphs are not only visually appealing and easy to understand, but they also allow you to obtain certain metrics and insights about the data. My goal with this page is to demonstrate how data analysis can help us better understand the world around us.

            ## Article Classifier

            The second feature is the "**Article Classifier**". This page allows users to upload an image of an article, which is then classified by an artificial intelligence model that I have trained. This model has been trained with TensorFlow and Keras, and is capable of identifying and classifying a variety of different articles. This feature is an example of how artificial intelligence can be used to automate tasks that would normally require human intervention.
        """)
    
    elif (language == 'Spanish') or (language == 'Español'):
    
        st.markdown("""
            # Bienvenidos

            Este proyecto es una demostración de mis habilidades en **análisis de datos**, **ciencia de datos** y el uso de técnicas de **inteligencia artificial**. He utilizado una variedad de tecnologías para desarrollar este proyecto, incluyendo **Pandas**, **TensorFlow**, **Keras**, y más. Mi objetivo es mostrar cómo estas herramientas pueden ser utilizadas para extraer conocimientos valiosos de los datos y resolver problemas complejos.

            ## Análisis de datos

            La primera funcionalidad que les quiero presentar es la página de "**Análisis de Datos**". En esta sección, podrán interactuar con diversas gráficas que he generado a partir de conjuntos de datos. Estas gráficas no sólo son visuales y fáciles de entender, sino que también permiten obtener ciertas métricas e insights sobre los datos. Mi objetivo con esta página es demostrar cómo el análisis de datos puede ayudarnos a entender mejor el mundo que nos rodea.

            ## Clasificador de artículos

            La segunda funcionalidad es el "**Clasificador de Artículos**". Esta página permite a los usuarios introducir una imagen de un artículo, que luego es clasificada por un modelo de inteligencia artificial que he entrenado. Este modelo ha sido entrenado con TensorFlow y Keras, y es capaz de identificar y clasificar una variedad de artículos diferentes. Esta funcionalidad es un ejemplo de cómo la inteligencia artificial puede ser utilizada para automatizar tareas que normalmente requerirían la intervención humana.
        """)

# %% Main

if __name__ == '__main__':

    select_language(state_manager)
    text_home(state_manager)