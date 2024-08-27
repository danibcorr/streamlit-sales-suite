import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtSmall
from tensorflow.keras import layers

from models.layers.data_augmentation import data_augmentation_layer


def build_model(
    dropout: float, fc_layers: list, num_classes: int, input_shape: tuple
) -> tf.keras.Model:
    """
    Build a classification model.

    Args:
        dropout (float): The dropout rate.
        fc_layers (list): The number of units in each fully connected layer.
        num_classes (int): The number of classes.
        input_shape (tuple, optional): The input shape.

    Returns:
        tf.keras.Model: The built classification model.
    """

    # Load the pre-trained model without the top layer
    base_model = ConvNeXtSmall(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # Freeze the layers of the pre-trained model due to limited memory
    for layer in base_model.layers:

        layer.trainable = False

    # Create an input layer to introduce the data
    inputs = layers.Input(shape=input_shape)

    # Use data augmentation
    x = data_augmentation_layer(inputs)

    # Use the features obtained from the pre-trained model
    x = base_model(x)

    # Reduce the dimensionality
    x = layers.GlobalAveragePooling2D()(x)

    # Use a few fully connected layers (MLPs)
    for fc in fc_layers:

        x = layers.Dense(fc, activation="gelu")(x)
        x = layers.Dropout(dropout)(x)

    # Get the output of the classification
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Get the model
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ClassifierModel")
