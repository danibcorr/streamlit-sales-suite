# %% Libraries

import tensorflow as tf
from models.layers.gadamw import GCAdamW

# %% Classes dataset

# String to number class
DICT_LABELS = {
    "cloth": 0, 
    "toys": 1, 
    "books": 2, 
    "jewelry": 3, 
    "perfumes": 4
}

# Number prediction to string class (Spanish)
PREDICTION_DICT_LABELS = {
    0: "Ropa", 
    1: "Juguete", 
    2: "Libro", 
    3: "Joyeria", 
    4: "Perfume"
}

# %% Parameters

# Items classifier parameters
PARAMS = {
    'num_classes': len(DICT_LABELS),
    'input_shape': (224, 224, 3),
    'FC Layers MLP': [512, 256],
    'Dropout MLP': 0.25,
    'Batch size': 32,
    'Num epochs': 20,
    'seed': 42
}

# Dataset division parameters
DATA_DIVISION = {
   'size_valid': 0.2,
   'size_test': 0.1
}

# Metrics to track loss and accuracy
METRICS = {
    'Compiled loss metric': tf.keras.losses.CategoricalFocalCrossentropy(),
    'Train loss metric': tf.keras.losses.CategoricalFocalCrossentropy(name='train_loss'),
    'Train accuracy metric': tf.keras.metrics.CategoricalAccuracy(name='train_accuracy'),
    'Valid loss metric': tf.keras.losses.CategoricalFocalCrossentropy(name='val_loss'),
    'Valid accuracy metric': tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
}

# Optimizer parameters
OPTIMIZER = {
    'Optimizer': GCAdamW,
    'Weight Decay': 1e-4
}

# Learning rate schedule parameters
LR_SCHEDULE = {
    'warmup_rate': 0.15,
    'lr_start': 1e-5,
    'lr_max': 1e-3
}