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


smooth = 1.
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
