# %% Libraries

import tensorflow as tf

# %% Class definition

@keras.utils.register_keras_serializable(package = 'GCAdamW')
class GCAdamW(tf.keras.optimizers.AdamW):

    """
    GCAdamW optimizer.

    This optimizer is a variant of the AdamW optimizer that applies gradient centralization.
    """

    def get_gradients(self, loss, params):

        """
        Compute the gradients of the loss with respect to the parameters.

        Args:
            loss (tf.Tensor): The loss tensor.
            params (list): The list of parameters.

        Returns:
            list: The list of gradients.
        """

        grads = []
        gradients = super().get_gradients(loss, params)

        for grad in gradients:
            
            grad_len = len(grad.shape)

            if grad_len > 1:

                axis = list(range(grad_len - 1))
                grad -= tf.math.reduce_mean(grad, axis = axis, keepdims = True)

            grads.append(grad)

        return grads

    def get_config(self):

        """
        Get the configuration of the optimizer.

        Returns:
            dict: A dictionary containing the configuration.
        """
        
        config = super().get_config()
        
        return config