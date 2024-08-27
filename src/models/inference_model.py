import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from PIL import Image

import streamlit as st
import tensorflow as tf
import numpy as np
from pickle import load
from deep_translator import GoogleTranslator

from config import config as cg
from models.layers.data_augmentation import data_augmentation_layer
from models.architecture import build_model
from web_functions.language_state import StateManager


language_manager = StateManager()
language = language_manager.language


@st.cache_resource
def load_model(model_path: str, num_classes: int) -> tf.keras.Model:

    model = build_model(
        dropout=cg.PARAMS["Dropout MLP"],
        fc_layers=cg.PARAMS["FC Layers MLP"],
        num_classes=cg.PARAMS["num_classes"],
        input_shape=cg.PARAMS["input_shape"],
    )

    model.load_weights(model_path)

    return model


@st.cache_data
def load_labels(labels_path: str) -> dict:

    with open(labels_path, "rb") as f:

        return load(f)


def image_processing(image: Image, col: st.columns) -> tf.Tensor:

    caption_title = (
        "Image loaded." if language in ["English", "Inglés"] else "Imagen cargada."
    )
    col.image(image, caption=caption_title, width=100)
    image = tf.keras.preprocessing.image.img_to_array(image, dtype="uint8")
    image = tf.image.resize(
        image, (cg.PARAMS["input_shape"][0], cg.PARAMS["input_shape"][1])
    )

    return tf.reshape(image, shape=(1, *cg.PARAMS["input_shape"]))


def make_prediction(
    model: tf.keras.Model, labels: dict, file: str, col2: st.columns
) -> str:

    image = Image.open(file)
    image = image_processing(image, col2)
    prediction = model.predict(image)
    prediction = np.argmax(prediction)
    og_predictions_string = labels[prediction]

    title = "Prediction:" if language in ["English", "Inglés"] else "Predicción: "

    if language in ["English", "Inglés"]:

        predictions_string = GoogleTranslator(source="es", target="en").translate(
            text=og_predictions_string
        )
        st.write(title, predictions_string)

    else:

        st.write(title, og_predictions_string)

    return og_predictions_string
