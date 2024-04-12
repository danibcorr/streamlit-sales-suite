# %% Librerias

import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# %% Parametros

st.set_page_config(
    page_title = "Clasificación de artículos",
    page_icon = "🔎"
)

st.title("🔎 Clasificación de artículos")

# %% Funciones

def load_model(path):

    return tf.keras.models.load_model(path)

def cargar_imagen(col):

    # Cargamos el archivo
    return col.file_uploader("Sube una imagen desde tu ordenador.", type=['png', 'jpg', 'jpeg'])

def procesamiento_imagen(imagen, col):

    # Convertir la imagen a escala de grises
    imagen = imagen.convert('L')

    # Mostrar la imagen
    col.image(imagen, caption = 'Imagen cargada.', width = 100)

    # Preprocesar la imagen para el modelo
    imagen = tf.keras.preprocessing.image.img_to_array(imagen)
    imagen = tf.image.resize(imagen, (28, 28))

    imagen = tf.expand_dims(imagen, axis=0)

    return imagen

# %% Main

if __name__ == '__main__':

    # Direccion del modelo que vamos a cargar
    path = 'models/mnist_model.keras'

    # Cargamos el modelo
    model = load_model(path = path)

    # Creamos las columnas
    col1, col2 = st.columns(2)

    # Cargamos la imagen que vamos a procesar y predecir su categoria
    archivo = cargar_imagen(col1)
    
    if archivo is not None:
        
        # Convertimos el archivo en una imagen
        imagen = Image.open(archivo) 

        # Procesamos la imagen
        imagen = procesamiento_imagen(imagen, col2)

        # Realizar la predicción
        predicciones = model.predict(imagen)

        # Mostrar las predicciones
        st.write('Predicción:', np.argmax(predicciones))