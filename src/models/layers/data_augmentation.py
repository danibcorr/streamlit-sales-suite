import tensorflow as tf
from tensorflow.keras import layers


def data_augmentation_layer(inputs: tf.Tensor, seed: int = 42) -> tf.Tensor:
    """
    Apply data augmentation to the input data.

    Args:
        inputs (tf.Tensor): The input data.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        tf.Tensor: The augmented input data.
    """

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(factor=0.4, seed=seed),
            layers.RandomZoom(height_factor=0.2, seed=seed),
            layers.RandomWidth(factor=0.2, seed=seed),
            layers.RandomTranslation(height_factor=0.2, width_factor=0.2, seed=seed),
            layers.GaussianNoise(stddev=0.1, seed=seed),
            layers.RandomBrightness(factor=0.2, seed=seed),
            layers.RandomContrast(factor=0.2, seed=seed),
        ],
        name="data_augmentation",
    )

    return data_augmentation(inputs)
