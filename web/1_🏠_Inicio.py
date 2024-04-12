# %% Librerias

import streamlit as st

# %% Parametros

st.set_page_config(
    page_title = "Inicio",
    page_icon = "🏠"
)

st.title("🏠 Inicio")

# %% Funciones

def texto_inicio():

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

    texto_inicio()