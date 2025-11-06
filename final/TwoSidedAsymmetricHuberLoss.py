import tensorflow as tf

# Corrected import for register_keras_serializable
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable() # Use the imported decorator directly
class TwoSidedAsymmetricHuberLoss(tf.keras.losses.Loss):
    """
    A custom two-sided asymmetric Huber loss function with value-dependent penalties.

    This loss penalizes:
    - Underpredictions (y_pred < y_true) more, especially when y_true is high (spikes).
    - Overpredictions (y_pred > y_true) more, especially when y_true is low (troughs/negative prices).
    """
    def __init__(self,
                 delta,
                 base_under_penalty_factor,
                 peak_penalty_scalar,
                 base_over_penalty_factor,
                 trough_penalty_scalar,
                 name="TwoSidedAsymmetricHuberLoss",
                 **kwargs):
        super().__init__(name=name)
        # Store all hyperparameters as attributes of the class
        self.delta = delta
        self.base_under_penalty_factor = base_under_penalty_factor
        self.peak_penalty_scalar = peak_penalty_scalar
        self.base_over_penalty_factor = base_over_penalty_factor
        self.trough_penalty_scalar = trough_penalty_scalar

    # This method defines the forward pass (how the loss is calculated)
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        abs_error = tf.math.abs(error)

        # Huber loss core
        quadratic_part = 0.5 * tf.math.square(error)
        linear_part = self.delta * abs_error - 0.5 * tf.math.square(self.delta)
        huber_loss_per_sample = tf.where(abs_error <= self.delta, quadratic_part, linear_part)

        # Masks for underpredictions and overpredictions
        # error > 0 means y_true > y_pred (underprediction)
        underprediction_mask = tf.cast(tf.math.greater(error, 0), tf.float32)
        # error <= 0 means y_true <= y_pred (overprediction or exact prediction)
        overprediction_mask = tf.cast(tf.math.less_equal(error, 0), tf.float32)

        # Dynamic penalty for UNDERPREDICTIONS (missing high spikes)
        # Penalty increases as y_true (normalized price) gets higher
        dynamic_under_penalty = self.base_under_penalty_factor + (y_true * self.peak_penalty_scalar)

        # Dynamic penalty for OVERPREDICTIONS (missing low spikes/troughs)
        # Penalty increases as y_true (normalized price) gets LOWER.
        # (1.0 - y_true) is large when y_true is small (near 0) and small when y_true is large (near 1).
        dynamic_over_penalty = self.base_over_penalty_factor + ((1.0 - y_true) * self.trough_penalty_scalar)

        # Apply the dynamic asymmetric weighting
        asymmetric_loss = (underprediction_mask * dynamic_under_penalty * huber_loss_per_sample) + \
                          (overprediction_mask * dynamic_over_penalty * huber_loss_per_sample)

        # Return the mean loss across the batch
        return tf.reduce_mean(asymmetric_loss)

    # This method is crucial for serialization: it tells Keras how to reconstruct the object
    # when loading a saved model. It should return a dictionary containing the arguments
    # needed to re-instantiate the class.
    def get_config(self):
        config = super().get_config() # Get the base config from the parent class
        config.update({
            "delta": self.delta,
            "base_under_penalty_factor": self.base_under_penalty_factor,
            "peak_penalty_scalar": self.peak_penalty_scalar,
            "base_over_penalty_factor": self.base_over_penalty_factor,
            "trough_penalty_scalar": self.trough_penalty_scalar,
        })
        return config


if __name__ == '__main__':
    print("Error: Don't call TwoSidedAsymmetricHuberLoss.py directly, import it.")

