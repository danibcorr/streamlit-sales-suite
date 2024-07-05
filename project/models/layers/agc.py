# %% Libraries

import tensorflow as tf

# %% Functions

def compute_norm(x: tf.Tensor, axis: int or list, keepdims: bool) -> tf.Tensor:

    """
    Compute the norm of a tensor.

    Args:
        x (tf.Tensor): The input tensor.
        axis (int or list of ints): The axis or axes to reduce.
        keepdims (bool): Whether to keep the reduced dimensions.

    Returns:
        tf.Tensor: The norm of the input tensor.
    """

    return tf.math.reduce_sum(x ** 2, axis = axis, keepdims = keepdims) ** 0.5

def unitwise_norm(x: tf.Tensor) -> tf.Tensor:

    """
    Compute the unit-wise norm of a tensor.

    Args:
        x (tf.Tensor): The input tensor.

    Returns:
        tf.Tensor: The unit-wise norm of the input tensor.
    """

    shape = len(x.get_shape())

    # Scalars and vectors
    if shape <= 1:  

        axis = None
        keepdims = False

    # Linear layers of shape IO or multihead linear
    elif shape in [2, 3]:  

        axis = 0
        keepdims = True

    # Conv kernels of shape HWIO
    elif shape == 4:  

        axis = [0, 1, 2]
        keepdims = True

    else:

        raise ValueError(f"Got a parameter with shape not in [1, 2, 4]! {x}")

    return compute_norm(x, axis, keepdims)

def adaptive_clip_grad(parameters: list or tf.Tensor, gradients: list or tf.Tensor, clip_factor: float = 0.01, eps: float = 1e-3) -> list or tf.Tensor:

    """
    Adaptive gradient clipping.

    Args:
        parameters (list of tf.Tensor): The model parameters.
        gradients (list of tf.Tensor): The gradients of the loss with respect to the parameters.
        clip_factor (float, optional): The clipping factor. Defaults to 0.01.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-3.

    Returns:
        list of tf.Tensor: The clipped gradients.
    """

    new_grads = []
    
    for (params, grads) in zip(parameters, gradients):
        
        p_norm = unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, eps) * clip_factor
        grad_norm = unitwise_norm(grads)
        clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        new_grads.append(new_grad)
    
    return new_grads