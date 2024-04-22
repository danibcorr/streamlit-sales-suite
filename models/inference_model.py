# %% Libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from pickle import load
from web_functions.language_state import StateManager
from models import config as cg
from models.layers.data_augmentation import data_augmentation_layer
from models.architecture import build_model

# %% Definitions for streamlit

if 'language' not in st.session_state:
    
    st.session_state.language = 'English'

state_manager = StateManager(language=st.session_state.language)

language = state_manager.get_language()

# %% Architecture

@st.cache_resource
def load_model(model_path: str, num_classes: int) -> tf.keras.Model:
    
    """
    Load a pre-trained model from a file.

    Args:
        model_path (str): Path to the model weights file.
        num_classes (int): Number of classes in the classification problem.

    Returns:
        tf.keras.Model: The loaded model.
    """

    # Create the model
    model = build_model(dropout = cg.PARAMS["Dropout MLP"], fc_layers = cg.PARAMS["FC Layers MLP"],
                        num_classes = cg.PARAMS["num_classes"], input_shape = cg.PARAMS["input_shape"])

    # Load the weights
    model.load_weights(model_path)

    return model

@st.cache_data
def load_labels(labels_path: str) -> dict:

    """
    Load labels from a file.

    Args:
        labels_path (str): Path to the labels file.

    Returns:
        dict: A dictionary of labels.
    """

    with open(labels_path, 'rb') as f:

        dict_labels = load(f)

    return dict_labels

# %% Data processing for input

def image_processing(image: Image, col: st.columns) -> tf.Tensor:

    """
    Process an input image for the model.

    Args:
        image (Image): The input image.
        col (st.columns): The Streamlit column to display the image.

    Returns:
        tf.Tensor: The preprocessed image tensor.
    """

    # Display the image.
    if (language == 'English') or (language == 'Inglés'):

        caption_title = 'Image loaded.'

    elif (language == 'Spanish') or (language == 'Español'):

        caption_title = 'Imagen cargada.'
    
    col.image(image, caption = caption_title, width = 100)

    # Preprocess image for the model
    image = tf.keras.preprocessing.image.img_to_array(image, dtype = 'uint8')
    image = tf.image.resize(image, (cg.PARAMS["input_shape"][0], cg.PARAMS["input_shape"][1]))

    # Apply preprocessing depending of the pre-trained model
    return tf.reshape(image, shape = (1, cg.PARAMS["input_shape"][0], cg.PARAMS["input_shape"][1], cg.PARAMS["input_shape"][-1]))

# %% Prediction

def make_prediction(model: tf.keras.Model, labels: dict, file: str, col2: st.columns) -> None:

    """
    Make a prediction using the model.

    Args:
        model (tf.keras.Model): The pre-trained model.
        labels (dict): A dictionary of labels.
        file (str): The input file path.
        col2 (st.columns): The Streamlit column to display the prediction.
    """

    # Convert the file to an image.
    image = Image.open(file) 

    # Process the image
    image = image_processing(image, col2)

    # Perform the prediction
    prediction = model.predict(image)
    prediction = np.argmax(prediction)
    predictions_string = labels[prediction]

    # Display predictions
    if (language == 'English') or (language == 'Inglés'):

        title = 'Prediction: '

    elif (language == 'Spanish') or (language == 'Español'):

        title = 'Predicción: '
    
    st.write(title, predictions_string)