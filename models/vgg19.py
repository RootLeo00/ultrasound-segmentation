#!/usr/bin/env python3
import os
from tabnanny import verbose
import tensorflow as tf
# from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
from PIL import Image as im
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
import random
import time
import sys
from tensorflow.keras import backend as K

from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, LeakyReLU, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam


#DEFINE THE MODEL

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


# Unet pre-trained model with VGG16 (weights: imagenet) 
def get_model(image_size, num_classes):
    # input: input_shape (height, width, channels) 
    # return model
    input_shape=image_size + (3,)

    #layer di preproce3ssing input
    inputs = keras.Input(shape=input_shape)
    preprocess_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    input_shape = input_shape
    # load the VGG19 network, ensuring the head FC layer sets are left off
    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=preprocess_input)

    #create top layer
    """ Encoder """
    s1 = vgg19.get_layer("block1_conv2").output         ## (512 x 512)
    s2 = vgg19.get_layer("block2_conv2").output         ## (256 x 256)
    s3 = vgg19.get_layer("block3_conv4").output         ## (128 x 128)
    s4 = vgg19.get_layer("block4_conv4").output         ## (64 x 64)

    """ Bridge """
    b1 = vgg19.get_layer("block5_conv4").output         ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(num_classes, 3, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    for layer in vgg19.layers:
        layer.trainable = False

    #compile in the train program to make it easy to test with various loss functions
    return model

