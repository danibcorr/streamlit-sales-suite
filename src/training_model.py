import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random
from pickle import dump

import mlflow
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, mixed_precision
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight, compute_class_weight
from keras_cv.layers import CutMix, ChannelShuffle, RandomCutout, RandAugment
from tqdm import tqdm
from loguru import logger

from models.architecture import build_model
from models.layers.agc import adaptive_clip_grad
from models.layers.lr_scheduler import warmupcosine_scheduler
from config import config as cg


# Log storage file
logger.add(cg.LOGS_TRAINIG_PATH, rotation="1 MB", compression="zip")

# Use mixed precision to reduce memory usage in GPU
# mixed_precision.set_global_policy("mixed_float16")


def mlflow_connection() -> None:
    """
    Establishes a connection with MLflow for experiment tracking and logging.

    This function sets the tracking server URI to http://127.0.0.1:8080 for logging.
    It creates a new MLflow experiment named "Item Classifier" and enables system metrics logging.

    Note:
        - Make sure MLflow is installed and running on the specified URI before calling this function.
        - Ensure that the tracking server URI is correctly configured for your MLflow setup.

    Returns:
        None
    """
    logger.info("Establishing connection with MLflow...")

    # Set the tracking server URI for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("Item Classifier")

    # Enable system metrics logging
    mlflow.enable_system_metrics_logging()


@tf.function
def train_step(
    model: tf.keras.Model,
    loss_function: tf.keras.losses.Loss,
    optimizer: tf.keras.optimizers.Optimizer,
    metric: tf.keras.metrics.Metric,
    class_weights: tf.Tensor,
    x: tf.Tensor,
    y: tf.Tensor,
) -> tf.Tensor:
    """
    Perform a single training step.

    Args:
        model (tf.keras.Model): The model to train.
        loss_function (tf.keras.losses.Loss): The loss function to use.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer to use.
        metric (tf.keras.metrics.Metric): The metric to track.
        class_weights (tf.Tensor): The class weights.
        x (tf.Tensor): The input data.
        y (tf.Tensor): The true labels.

    Returns:
        tf.Tensor: The weighted loss value.
    """

    with tf.GradientTape() as tape:

        # Gather the weights corresponding to the classes of the true labels
        weights = tf.gather(class_weights, tf.argmax(y, axis=-1))

        # Forward pass
        logits = model(x, training=True)

        # Calculate the loss
        loss_value = loss_function(y, logits) * weights

        # Scale the loss for mixed precision training
        scaled_loss = optimizer.scale_loss(loss_value)

    # Calculate the gradients
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)

    # Unscale the gradients
    # gradients = optimizer.get_unscaled_gradients(scaled_gradients)

    # Clip the gradients
    # clipped_gradients = adaptive_clip_grad(parameters = model.trainable_variables, gradients = scaled_gradients)

    # Apply the gradients
    optimizer.apply_gradients(zip(scaled_gradients, model.trainable_variables))

    # Update the metric
    metric.update_state(y, logits)

    # Take the mean of the weighted losses
    return tf.reduce_mean(loss_value)


@tf.function
def val_step(
    model: tf.keras.Model,
    loss_fn: tf.keras.losses.Loss,
    metric: tf.keras.metrics.Metric,
    x: tf.Tensor,
    y: tf.Tensor,
) -> tf.Tensor:
    """
    Perform a single validation step.

    Args:
        model (tf.keras.Model): The model to validate.
        loss_fn (tf.keras.losses.Loss): The loss function to use.
        metric (tf.keras.metrics.Metric): The metric to track.
        x (tf.Tensor): The input data.
        y (tf.Tensor): The true labels.

    Returns:
        tf.Tensor: The loss value.
    """

    # Perform a forward pass to get the model's predictions
    val_logits = model(x, training=False)

    # Calculate the loss using the provided loss function
    loss_value = loss_fn(y, val_logits)

    # Update the metric with the true labels and predicted logits
    metric.update_state(y, val_logits)

    # Return the calculated loss value
    return loss_value


def combine_dictionaries() -> dict:
    """
    Combines multiple dictionaries into one.

    Returns:
        dict: A dictionary containing the combined key-value pairs from PARAMS, DATA_DIVISION,
              METRICS, OPTIMIZER, and LR_SCHEDULE dictionaries.
    """

    # Initialize an empty dictionary to store the combined key-value pairs
    combined_dict = {}

    # Update the combined dictionary with the key-value pairs from each dictionary
    combined_dict.update(cg.PARAMS)
    combined_dict.update(cg.DATA_DIVISION)
    combined_dict.update(cg.METRICS)
    combined_dict.update(cg.OPTIMIZER)
    combined_dict.update(cg.LR_SCHEDULE)

    # Return the combined dictionary
    return combined_dict


def training_valid_loop(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    class_weights: tf.Tensor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> None:
    """
    Train and validate the model using a custom training loop.

    Args:
        model (tf.keras.Model): The model to train.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer to use for training.
        class_weights (tf.Tensor): The class weights for training.
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The training labels.
        X_valid (np.ndarray): The validation data.
        y_valid (np.ndarray): The validation labels.

    Returns:
        None
    """

    # Create instances of additional data augmentation techniques
    cutmix_layer = CutMix(alpha=1.0, seed=cg.PARAMS["seed"])
    channel_shuffle_layer = ChannelShuffle(groups=3, seed=cg.PARAMS["seed"])
    cg.PARAMS["Data Augmentation"] = [cutmix_layer, channel_shuffle_layer]
    num_additional_aug = len(cg.PARAMS["Data Augmentation"])

    # Combine all the params to be stored in MLflow
    params = combine_dictionaries()

    # Create instances for the model parameters
    num_epochs = cg.PARAMS["Num epochs"]
    batch_size = cg.PARAMS["Batch size"]
    train_loss = cg.METRICS["Train loss metric"]
    train_accuracy = cg.METRICS["Train accuracy metric"]
    val_loss = cg.METRICS["Valid loss metric"]
    val_accuracy = cg.METRICS["Valid accuracy metric"]

    # Start an MLFlow run
    with mlflow.start_run():

        # Log the parameters
        mlflow.log_params(params)

        # Custom training loop
        for epoch in range(num_epochs):

            logger.info(f"Training model, epoch {epoch}...")

            with tqdm(
                total=X_train.shape[0] + X_valid.shape[0],
                desc=f"Epoch: {epoch + 1}/{num_epochs}, lr = {optimizer.learning_rate.numpy()}",
            ) as pbar:

                # Run training loop iteration on batches
                for idx in range(0, X_train.shape[0], batch_size):

                    # Take the samples from the selected indices
                    X_batch_train = X_train[idx : idx + batch_size]
                    y_batch_train = y_train[idx : idx + batch_size]

                    # Convert images and labels to tensors
                    X_batch_train_tensor = tf.convert_to_tensor(
                        X_batch_train, dtype=tf.float32
                    )
                    y_batch_train_tensor = tf.convert_to_tensor(
                        y_batch_train, dtype=tf.float32
                    )

                    # Prepare the data in the format that CutMix expects
                    data = {
                        "images": X_batch_train_tensor,
                        "labels": y_batch_train_tensor,
                    }

                    # Generate a random number
                    num_layers = np.random.randint(num_additional_aug)

                    # Apply the layers randomly
                    if num_layers > 0:
                        data = cutmix_layer(data)
                    if num_layers > 1:
                        data = channel_shuffle_layer(data)

                    # Apply 1 train step
                    loss_value = train_step(
                        model,
                        train_loss,
                        optimizer,
                        train_accuracy,
                        class_weights,
                        data["images"],
                        data["labels"],
                    )

                    # Update the progress bar
                    pbar.set_postfix(
                        {
                            "Train Loss": f"{loss_value:.4f}",
                            "Train Accuracy": f"{train_accuracy.result():.4f}",
                        }
                    )
                    pbar.update(batch_size)

                # Run validation loop iteration on batches
                for idx in range(0, X_valid.shape[0], batch_size):

                    # Take the samples from the selected indices
                    X_batch_valid = X_valid[idx : idx + batch_size]
                    y_batch_valid = y_valid[idx : idx + batch_size]

                    # Apply 1 validation step
                    loss_value_valid = val_step(
                        model, val_loss, val_accuracy, X_batch_valid, y_batch_valid
                    )

                    # Update the progress bar
                    pbar.set_postfix(
                        {
                            "Train Loss": f"{loss_value:.4f}",
                            "Train Accuracy": f"{train_accuracy.result():.4f}",
                            "Valid Loss": f"{loss_value_valid:.4f}",
                            "Valid Accuracy": f"{val_accuracy.result():.4f}",
                        }
                    )
                    pbar.update(batch_size)

                # Log data in MLFlow
                mlflow.log_metrics(
                    {
                        "Epoch": epoch,
                        "Train loss": loss_value,
                        "Train accuracy": train_accuracy.result(),
                        "Valid loss": loss_value_valid,
                        "Valid accuracy": val_accuracy.result(),
                        "Learning rate": optimizer.learning_rate.numpy(),
                    },
                    step=epoch,
                )

                # Reset metrics at the end of each epoch
                train_accuracy.reset_state()
                val_accuracy.reset_state()

    # Save the model's weights
    logger.info("Saving model, weights only and keras model")
    model.save_weights(cg.MODEL_PATH)
    model.save(cg.MODEL_PATH_KERAS)
    tf.keras.backend.clear_session()


def seed_init(seed: int) -> None:
    """
    Initialize random seeds for NumPy, TensorFlow, and Python's random module.

    This function ensures reproducibility of results by setting the same seed value
    for all random number generators used in the program.

    Args:
        seed (int): The seed value to use for random initialization.

    Returns:
        None
    """
    logger.info("Seed initiation to allow reproducibility of the code...")

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Set the seed for TensorFlow's random number generator
    tf.random.set_seed(seed)

    # Set the seed for Python's built-in random module
    random.seed(seed)


def load_dataset(path: str, pre_trained: bool, input_shape: tuple) -> tuple:
    """
    Load and preprocess the dataset.

    This function loads images from a directory, resizes them to the desired shape,
    and converts them to the required format for a pre-trained model.

    Args:
        path (str): The path to the dataset.
        pre_trained (bool): Whether the dataset is pre-trained.
        input_shape (tuple): The desired shape of the input images.

    Returns:
        tuple: A tuple containing the loaded images and their corresponding labels.
    """
    logger.info("Loading the dataset...")

    images = []
    labels = []

    # Iterate over all subfolders in the main folder
    for folder in os.listdir(path):

        folder_path = os.path.join(path, folder)

        # Check if it's a real subfolder
        if os.path.isdir(folder_path):

            # Iterate over all files in the subfolder
            for filename in os.listdir(folder_path):

                # Load the image
                img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_COLOR)

                if img is not None:

                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Resize the image to the desired resolution
                    img_resized = cv2.resize(img, (input_shape[0], input_shape[1]))

                    # Save the image and its corresponding label
                    images.append(img_resized)

                    # Use the subfolder name as the label
                    labels.append(folder)

    # Convert the list to a NumPy array
    images = np.array(images)

    # Reshape the array to the desired format for the pre-trained model
    images = images.reshape(
        images.shape[0], input_shape[0], input_shape[1], input_shape[-1]
    ).astype("uint8")

    return images, np.array(labels)


def class_weights_calculation(labels: np.ndarray, y_train: np.ndarray) -> tf.Tensor:
    """
    Calculate class weights to address class imbalance in the dataset.

    This function calculates the class weights using the 'balanced' method, which
    assigns more weight to classes with lower frequencies in the dataset.

    Args:
        labels (np.ndarray): The labels of the dataset.
        y_train (np.ndarray): The one-hot encoded labels of the training set.

    Returns:
        tf.Tensor: The class weights.
    """
    logger.info(
        "Calculating the relevance of the samples in the data set for the assignment of weights..."
    )

    # Calculate the class weights using the 'balanced' method
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=np.argmax(y_train, axis=-1),
    )

    # Convert the class weights to a dictionary
    class_weights = dict(enumerate(class_weights))

    # Convert the class weights to a TensorFlow tensor with float16 data type
    class_weights = tf.constant(
        [class_weights[i] for i in range(len(class_weights))], dtype=tf.float32
    )

    return class_weights


def dataset_split(data: np.ndarray, labels: np.ndarray, num_classes: int) -> tuple:
    """
    Split the dataset into train, validation, and test sets.

    This function splits the dataset into three parts: training, validation, and testing.
    It uses stratified splitting to ensure that the class distribution is preserved in each set.

    Args:
        data (np.ndarray): The dataset.
        labels (np.ndarray): The labels of the dataset.
        num_classes (int): The number of classes.

    Returns:
        tuple: A tuple containing train, validation, and test datasets and labels.
    """
    logger.info("Division of the dataset...")

    # Get the sizes of the validation and test sets from the configuration
    size_valid = cg.DATA_DIVISION["size_valid"]
    size_test = cg.DATA_DIVISION["size_test"]

    # Split the dataset into training and validation + test sets
    X_train, X_valid_test, y_train, y_valid_test = train_test_split(
        data,
        labels,
        test_size=size_valid + size_test,
        random_state=cg.PARAMS["seed"],
        stratify=labels,
    )

    # Split the validation + test set into validation and test sets
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid_test,
        y_valid_test,
        test_size=size_test / (size_valid + size_test),
        random_state=cg.PARAMS["seed"],
        stratify=y_valid_test,
    )

    # Convert labels to one-hot encoding
    y_train = tf.one_hot(y_train, depth=num_classes)
    y_valid = tf.one_hot(y_valid, depth=num_classes)
    y_test = tf.one_hot(y_test, depth=num_classes)

    # Print the shapes of the datasets
    print(f"Train shape: {X_train.shape}")
    print(f"Valid shape: {X_valid.shape}")
    print(f"Test shape: {X_test.shape}\n")

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def main(
    dataset_path: str,
    dict_labels_path: str,
) -> None:
    """
    Main function.

    This function is the entry point of the program. It performs the following tasks:
    1. Establishes an MLflow connection.
    2. Saves the dictionary of labels.
    3. Initializes the seeds for reproducibility.
    4. Loads the dataset and performs additional image processing.
    5. Splits the dataset into training, validation, and testing sets.
    6. Calculates class weights to address class imbalance.
    7. Creates and compiles the model.
    8. Trains and saves the model.

    Args:
        dataset_path (str): The path to the dataset.
        dict_labels_path (str, optional): The path to save the dictionary of labels. Defaults to "./models/trained_model/dict_labels.pkl".
    """

    # Stablish MLflow connection
    mlflow_connection()

    # Save the dictionary
    with open(dict_labels_path, "wb") as f:

        dump(cg.PREDICTION_DICT_LABELS, f)

    # Initialize the seeds
    seed_init(cg.PARAMS["seed"])

    # Load the dataset and perform additional image processing
    data, labels = load_dataset(
        path=dataset_path, pre_trained=True, input_shape=cg.PARAMS["input_shape"]
    )

    # Map the list values to the dictionary values
    labels = np.array([cg.DICT_LABELS[i] for i in labels])

    # Print the shapes
    print(f"\nData shape: {data.shape}\nLabels shape: {labels.shape}\n")

    # Dataset split
    X_train, y_train, X_valid, y_valid, X_test, y_test = dataset_split(
        data, labels, cg.PARAMS["num_classes"]
    )
    cg.PARAMS["Train shape"] = X_train.shape
    cg.PARAMS["Valid shape"] = X_valid.shape
    cg.PARAMS["Test shape"] = X_test.shape

    # Add weights to the classes to improbe the performance in the imbalance dataset
    class_weights = class_weights_calculation(labels, y_train)

    # Create the model
    model = build_model(
        dropout=cg.PARAMS["Dropout MLP"],
        fc_layers=cg.PARAMS["FC Layers MLP"],
        num_classes=cg.PARAMS["num_classes"],
        input_shape=cg.PARAMS["input_shape"],
    )

    # Compile the projection model
    scheduled_lrs = warmupcosine_scheduler(
        X_train,
        cg.PARAMS["Batch size"],
        cg.PARAMS["Num epochs"],
        cg.LR_SCHEDULE["warmup_rate"],
        cg.LR_SCHEDULE["lr_start"],
        cg.LR_SCHEDULE["lr_max"],
    )
    optimizer = cg.OPTIMIZER["Optimizer"](
        learning_rate=scheduled_lrs, weight_decay=cg.OPTIMIZER["Weight Decay"]
    )
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(optimizer, loss=cg.METRICS["Compiled loss metric"])
    print(model.summary())

    # Train and save the model
    training_valid_loop(
        model, optimizer, class_weights, X_train, y_train, X_valid, y_valid
    )


if __name__ == "__main__":

    main(cg.IMGS_DATASET_PATH, cg.LABELS_PATH)
