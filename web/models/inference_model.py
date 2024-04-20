# %% Libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from pickle import load
from tensorflow.keras.applications.convnext import preprocess_input
from tensorflow.keras.applications import ConvNeXtSmall
from . import config as cg

# %% Architecture

def data_augmentation_layer(inputs, seed = 42):

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed = seed),
            layers.RandomRotation(factor = 0.4, seed = seed),
            layers.RandomZoom(height_factor = 0.2, seed = seed),
            layers.RandomWidth(factor = 0.2, seed = seed),
            layers.RandomTranslation(height_factor = 0.2, width_factor = 0.2, seed = seed),
            layers.GaussianNoise(stddev=0.1, seed=seed),
            layers.RandomBrightness(factor = 0.2, seed=seed),
            layers.RandomContrast(factor = 0.2, seed=seed)
        ],
        name = "data_augmentation"
    )

    return data_augmentation(inputs)


@st.cache_resource
def build_model(model_path, num_classes):

    # Loads the ResNet model without the top layer
    base_model = ConvNeXtSmall(weights = 'imagenet', include_top = False, input_shape = cg.input_shape)

    # We freeze the layers of the pre-trained model
    for layer in base_model.layers:

        layer.trainable = False

    # We create a layer to input the data
    inputs = layers.Input(shape = cg.input_shape)
    
    # We use data augmentation
    x = data_augmentation_layer(inputs)
    
    # We use the features obtained from the pre-trained model
    x = base_model(x)
    
    # Reduce dimensionality
    x = layers.GlobalAveragePooling2D()(x)

    # We use a pair of MLPs
    for fc in cg.fc_layers:

        x = layers.Dense(fc, activation = 'gelu')(x) 
        x = layers.Dropout(cg.dropout)(x)

    # We get the output
    outputs = layers.Dense(num_classes, activation = "softmax")(x)

    # Get the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ClassifierModel")

    # We load the weights
    model.load_weights(model_path)

    return model

@st.cache_data
def load_labels(labels_path):

    with open(labels_path, 'rb') as f:
        
        dict_labels = load(f)

    return dict_labels

# %% Data processing for input

def image_processing(image, col):

    # Display the image.
    col.image(image, caption = 'Imagen cargada.', width = 100)

    # Preprocess image for the model
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.image.resize(image, (cg.input_shape[0], cg.input_shape[1]))

    # Apply preprocessing depending of the pre-trained model
    return tf.reshape(image, shape = (1, cg.input_shape[0], cg.input_shape[1], cg.input_shape[-1]))

# %% Prediction

def make_prediction(model, labels, file, col2):

    # We convert the file to an image.
    image = Image.open(file) 

    # Process the image
    image = image_processing(image, col2)

    # Perform the prediction
    prediction = model.predict(image)
    prediction = np.argmax(prediction)
    predictions_string = labels[prediction]

    # Display predictions
    st.write('Predicción:', predictions_string)