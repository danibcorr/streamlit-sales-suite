import tensorflow as tf
import numpy as np


@tf.keras.utils.register_keras_serializable(package="WarmUpCosine")
class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Warm-up cosine learning rate schedule.

    Args:
        lr_start (float): The initial learning rate.
        lr_max (float): The maximum learning rate.
        warmup_steps (int): The number of steps for warm-up.
        total_steps (int): The total number of steps for training.

    Attributes:
        lr_start (float): The initial learning rate.
        lr_max (float): The maximum learning rate.
        warmup_steps (int): The number of steps for warm-up.
        total_steps (int): The total number of steps for training.
        pi (tf.Tensor): A constant representing pi.
    """

    def __init__(self, lr_start, lr_max, warmup_steps, total_steps):

        super().__init__()

        self.lr_start = lr_start
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        """
        Compute the learning rate at a given step.

        Args:
            step (int): The current step.

        Returns:
            tf.Tensor: The learning rate at the given step.
        """

        if self.total_steps < self.warmup_steps:

            raise ValueError(
                f"Total number of steps {self.total_steps} must be"
                + f"larger or equal to warmup steps {self.warmup_steps}."
            )

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        )

        learning_rate = 0.5 * self.lr_max * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:

            if self.lr_max < self.lr_start:

                raise ValueError(
                    f"lr_start {self.lr_start} must be smaller or"
                    + f"equal to lr_max {self.lr_max}."
                )

            slope = (self.lr_max - self.lr_start) / self.warmup_steps

            warmup_rate = slope * tf.cast(step, tf.float32) + self.lr_start

            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )

        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )

    def get_config(self):
        """
        Get the configuration of the learning rate schedule.

        Returns:
            dict: A dictionary containing the configuration.
        """

        config = {
            "lr_start": self.lr_start,
            "lr_max": self.lr_max,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }

        return config


def warmupcosine_scheduler(
    X_train: np.ndarray,
    batch_size: int,
    num_epochs: int,
    warmup_rate: float = 0.15,
    lr_start: float = 1e-5,
    lr_max: float = 1e-3,
) -> WarmUpCosine:
    """
    Create a warm-up cosine learning rate schedule.

    Args:
        X_train (np.ndarray): The training data.
        batch_size (int): The batch size.
        num_epochs (int): The number of epochs.
        warmup_rate (float, optional): The warm-up rate. Defaults to 0.15.
        lr_start (float, optional): The initial learning rate. Defaults to 1e-5.
        lr_max (float, optional): The maximum learning rate. Defaults to 1e-3.

    Returns:
        WarmUpCosine: A warm-up cosine learning rate schedule.
    """

    # Get the total number of steps for training.
    total_steps = int((X_train.shape[0] / batch_size) * num_epochs)

    # Calculate the number of steps for warmup.
    warmup_steps = int(total_steps * warmup_rate)

    # Initialize the warmup cosine schedule.
    scheduled_lrs = WarmUpCosine(
        lr_start=lr_start,
        lr_max=lr_max,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    return scheduled_lrs
