#!/usr/bin/env python3
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


def precision_m(y_true, y_pred):
    # print(type(y_true))
    # print(type(y_pred))
    # print(y_true.shape)
    # print(y_pred.shape)
    # exit()
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
