import os
import sys
from keras.metrics import MeanIoU
import tensorflow as tf
# from IPython.display import Image, display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from params import pred_dir, NUM_CLASSES
import numpy as np

# TODO: copiare plot dei grafici da script (fa dei plot più carini)


smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # Input 'y' of 'Mul' Op has type float32 that does not match type uint8 of argument 'x'.
    y_true_f = tf.cast(y_pred_f, tf.float32)  # convert from uint8 to float32
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

