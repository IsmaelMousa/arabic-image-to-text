import tensorflow as tf
import numpy as np
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
from tensorflow.keras.backend import ctc_decode

def decode_batch_predictions(pred, lookup):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(lookup(res)).numpy().decode("utf-8")
        output_text.append(res.replace("[UNK]", ""))

    return output_text


def predict(image, model, lookup, display=True):
    preds = model.predict(tf.convert_to_tensor(image))
    pred_texts = decode_batch_predictions(preds, lookup)

    pred_text = pred_texts[0]
    pred_text = arabic_reshaper.reshape(pred_text)
    if display:
        pred_text = get_display(pred_text)

    return pred_text