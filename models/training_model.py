# %% Libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from pickle import dump
import mlflow
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision
import pandas as pd
import random
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras_cv.layers import CutMix, ChannelShuffle, RandomCutout, RandAugment
from tqdm import tqdm

from architecture import build_model
from layers.agc import adaptive_clip_grad
from layers.lr_scheduler import warmupcosine_scheduler
import config as cg

# %% Let's use mixed precision to reduce memory consumption.

mixed_precision.set_global_policy('mixed_float16')

# %% Functions

def mlflow_connection():

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

    # Set the tracking server URI for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("Item Classifier")

    # Enable system metrics logging
    mlflow.enable_system_metrics_logging()


@tf.function
def train_step(model: tf.keras.Model, loss_function: tf.keras.losses.Loss, optimizer: tf.keras.optimizers.Optimizer,
               metric: tf.keras.metrics.Metric, class_weights: tf.Tensor, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    
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
        
        logits = model(x, training=True)
        loss_value = loss_function(y, logits) * weights
        scaled_loss = optimizer.get_scaled_loss(loss_value)

    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    clipped_gradients = adaptive_clip_grad(parameters=model.trainable_variables, gradients=gradients)
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
    
    metric.update_state(y, logits)

    # Take the mean of the weighted losses
    return tf.reduce_mean(loss_value)

@tf.function
def val_step(model: tf.keras.Model, loss_fn: tf.keras.losses.Loss, metric: tf.keras.metrics.Metric,
             x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:

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

    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    metric.update_state(y, val_logits)

    return loss_value

def combine_dictionary() -> dict:

    """
    Combines multiple dictionaries into one.

    Returns:
    dict: A dictionary containing the combined key-value pairs from PARAMS, DATA_DIVISION,
          METRICS, OPTIMIZER, and LR_SCHEDULE dictionaries.
    """
    combined_dict = {}
    
    combined_dict.update(cg.PARAMS)
    combined_dict.update(cg.DATA_DIVISION)
    combined_dict.update(cg.METRICS)
    combined_dict.update(cg.OPTIMIZER)
    combined_dict.update(cg.LR_SCHEDULE)

    return combined_dict

def traininig_valid_loop(model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, class_weights: tf.Tensor,
                         X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray) -> None:

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

    # Create several instances of additional data augmentation techniques
    cutmix_layer = CutMix(alpha = 1.0, seed = cg.PARAMS["seed"])
    channel_shuffle_layer = ChannelShuffle(groups = 3, seed = cg.PARAMS["seed"])
    cg.PARAMS["Data Augmentation"] = [cutmix_layer, channel_shuffle_layer]
    num_additional_aug = len(cg.PARAMS["Data Augmentation"])

    # Combine all the params to be stored in MLflow
    params = combine_dictionary()

    # Create the instances for the model parameters
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

            with tqdm(total = X_train.shape[0] + X_valid.shape[0], desc = f"Epoch: {epoch + 1}/{num_epochs}, lr = {optimizer.learning_rate.numpy()}") as pbar:

                # Run training loop iteration on batches
                for idx in range(0, X_train.shape[0], batch_size):

                    # Take the samples from the selected indices
                    X_batch_train = X_train[idx:idx+batch_size]
                    y_batch_train = y_train[idx:idx+batch_size]

                    # Prepare the data in the format that CutMix expects
                    data = {"images": X_batch_train, "labels": y_batch_train}

                    # Generate a random number
                    num_layers = np.random.randint(num_additional_aug)

                    # Apply the layers randomly
                    if num_layers > 0:

                        data = cutmix_layer(data)

                    if num_layers > 1:

                        data = channel_shuffle_layer(data)

                    # Apply 1 train step
                    loss_value = train_step(model, train_loss, optimizer, train_accuracy, class_weights, data["images"], data["labels"])

                    # Update the progress bar
                    pbar.set_postfix({'Train Loss': f'{loss_value:.4f}', 'Train Accuracy': f'{train_accuracy.result():.4f}'})
                    pbar.update(batch_size)

                # Run validation loop iteration on batches
                for idx in range(0, X_valid.shape[0], batch_size):

                    # Take the samples from the selected indices
                    X_batch_valid = X_valid[idx:idx+batch_size]
                    y_batch_valid = y_valid[idx:idx+batch_size]
                    
                    # Apply 1 validation step
                    loss_value_valid = val_step(model, val_loss, val_accuracy, X_batch_valid, y_batch_valid)

                    # Update the progress bar
                    pbar.set_postfix({'Train Loss': f'{loss_value:.4f}', 'Train Accuracy': f'{train_accuracy.result():.4f}',
                                      'Valid Loss': f'{loss_value_valid:.4f}', 'Valid Accuracy': f'{val_accuracy.result():.4f}'})
                
                    pbar.update(batch_size)

                # Log data in MLFlow
                mlflow.log_metrics({
                    "Epoch": epoch,
                    "Train loss": loss_value,
                    "Train accuracy": train_accuracy.result(),
                    "Valid loss": loss_value_valid,
                    "Valid accuracy": val_accuracy.result(),
                    "Learning rate": optimizer.learning_rate.numpy()
                }, step=epoch)


                # Reset metrics at the end of each epoch
                train_accuracy.reset_states()
                val_accuracy.reset_states()

    # Save the model's weights 
    model.save_weights('./trained_model/item_classifier')

def seed_init(seed: int) -> None:
    
    """
    Initialize random seeds for NumPy, TensorFlow, and Python's random module.

    Args:
        seed (int): The seed value to use for random initialization.

    Returns:
        None
    """

    np.random.seed(seed)  
    tf.random.set_seed(seed)
    random.seed(seed)

def load_dataset(path: str, pre_trained: bool, input_shape: tuple) -> tuple:
    
    """
    Load and preprocess the dataset.

    Args:
        path (str): The path to the dataset.
        pre_trained (bool): Whether the dataset is pre-trained.
        input_shape (tuple): The desired shape of the input images.

    Returns:
        tuple: A tuple containing the loaded images and their corresponding labels.
    """

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
    images = images.reshape(images.shape[0], input_shape[0], input_shape[1], input_shape[-1]).astype('uint8')

    return images, np.array(labels)

def class_weights_calculation(labels: np.ndarray, y_train: np.ndarray) -> tf.Tensor:

    """
    Calculate class weights to address class imbalance in the dataset.

    Args:
        labels (np.ndarray): The labels of the dataset.
        y_train (np.ndarray): The one-hot encoded labels of the training set.

    Returns:
        tf.Tensor: The class weights.
    """

    # Calculate the class weights and reduce the problem of class imbalance in the data
    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(labels), y = np.argmax(y_train, axis = -1))
    class_weights = dict(enumerate(class_weights))
    class_weights = tf.constant([class_weights[i] for i in range(len(class_weights))], dtype = tf.float16)

    return class_weights

def dataset_split(data: np.ndarray, labels: np.ndarray, num_classes: int) -> tuple:
    
    """
    Split the dataset into train, validation, and test sets.

    Args:
        data (np.ndarray): The dataset.
        labels (np.ndarray): The labels of the dataset.
        num_classes (int): The number of classes.

    Returns:
        tuple: A tuple containing train, validation, and test datasets and labels.
    """

    # Create instances
    size_valid = cg.DATA_DIVISION["size_valid"] 
    size_test = cg.DATA_DIVISION["size_test"]

    # Split dataset
    X_train, X_valid_test, y_train, y_valid_test = train_test_split(data, labels, test_size = size_valid + size_test,
                                                                    random_state = cg.PARAMS["seed"], stratify = labels)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size = size_test / (size_valid + size_test), 
                                                        random_state = cg.PARAMS["seed"], stratify = y_valid_test)

    # Convert labels to One-hot encoding
    y_train = tf.one_hot(y_train, depth = num_classes)
    y_valid = tf.one_hot(y_valid, depth = num_classes)
    y_test = tf.one_hot(y_test, depth = num_classes)

    # Print shape datasets
    print(f"Train shape: {X_train.shape}")
    print(f"Valid shape: {X_valid.shape}")
    print(f"Test shape: {X_test.shape}\n")

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def main(dataset_path: str, dict_labels_path: str = "./models/trained_model/dict_labels.pkl") -> None:

    """
    Main function.
    """

    # Stablish MLflow connection
    mlflow_connection()

    # Save the dictionary
    with open(dict_labels_path, 'wb') as f:
        
        dump(cg.PREDICTION_DICT_LABELS, f)

    # Initialize the seeds
    seed_init(cg.PARAMS['seed'])

    # Load the dataset and perform additional image processing
    data, labels = load_dataset(path = dataset_path, pre_trained = True, input_shape = cg.PARAMS["input_shape"])

    # Map the list values to the dictionary values
    labels = np.array([cg.DICT_LABELS[i] for i in labels])

    # Print the shapes
    print(f"\nData shape: {data.shape}\nLabels shape: {labels.shape}\n")

    # Dataset split
    X_train, y_train, X_valid, y_valid, X_test, y_test = dataset_split(data, labels, cg.PARAMS["num_classes"])
    cg.PARAMS["Train shape"] = X_train.shape
    cg.PARAMS["Valid shape"] = X_valid.shape
    cg.PARAMS["Test shape"] = X_test.shape

    # Add weights to the classes to improbe the performance in the imbalance dataset
    class_weights = class_weights_calculation(labels, y_train)

    # Create the model
    model = build_model(dropout = cg.PARAMS["Dropout MLP"], fc_layers = cg.PARAMS["FC Layers MLP"],
                        num_classes = cg.PARAMS["num_classes"], input_shape = cg.PARAMS["input_shape"])

    # Compile the projection model
    scheduled_lrs = warmupcosine_scheduler(X_train, cg.PARAMS["Batch size"], cg.PARAMS["Num epochs"], cg.LR_SCHEDULE["warmup_rate"], cg.LR_SCHEDULE["lr_start"], cg.LR_SCHEDULE["lr_max"])
    optimizer = cg.OPTIMIZER['Optimizer'](learning_rate = scheduled_lrs, weight_decay = cg.OPTIMIZER['Weight Decay'])
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(optimizer, loss = cg.METRICS["Compiled loss metric"])
    print(model.summary())

    # Train and save the model
    traininig_valid_loop(model, optimizer, class_weights, X_train, y_train, X_valid, y_valid)

# %% Main

if __name__ == '__main__':

    # Dataset path
    dataset_path = './datasets/items/'
    main(dataset_path)