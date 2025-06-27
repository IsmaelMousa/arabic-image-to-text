from utils import CTCLayer
import tensorflow as tf
import keras


def model_loader(path, predict=True):
    model = tf.keras.models.load_model(path, custom_objects={'CTCLayer': CTCLayer})

    if  predict:
        prediction_model = keras.models.Model(
            inputs=model.inputs[0],
            outputs=model.get_layer("dense2").output
        )

        return prediction_model

    return model


def lookup_loader(chr_path, num_path):
    char_to_num = tf.keras.models.load_model(chr_path)
    num_to_char = tf.keras.models.load_model(num_path)

    return char_to_num, num_to_char

