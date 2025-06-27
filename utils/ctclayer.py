import tensorflow as tf
import keras


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        """
        Defines the forward pass of the layer where the computation happens
        :param y_true: ground truth labels
        :param y_pred: model predictions
        :return: Returns the predictions unchanged
        """

        # Gets the batch size (number of samples in the batch) from y_true and converts it to int64.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")

        # Gets the sequence length of the predictions (second dimension of y_pred) and converts to int64.
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")

        # Gets the sequence length of the labels (second dimension of y_true) and converts to int64.
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        # Creates a tensor of shape (batch_size, 1) where each element is the input sequence length.
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        # Creates a tensor of shape (batch_size, 1) where each element is the label sequence length.
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred
