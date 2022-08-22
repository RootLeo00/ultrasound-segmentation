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
def TL_unet_model(input_shape, NUM_CLASSES):
    # input: input_shape (height, width, channels) 
    # return model

    #layer di preproce3ssing input
    inputs = keras.Input(shape=input_shape)
    preprocess_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    input_shape = input_shape
    # load the VGG19 network, ensuring the head FC layer sets are left off
    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=preprocess_input)

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
    outputs = Conv2D(NUM_CLASSES, 3, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    for layer in vgg19.layers:
        layer.trainable = False
    return model

# # Create new model on top
#     # the bridge (exclude the last maxpooling layer in VGG19) 
#     bridge = base_VGG.get_layer("block5_conv3").output
#     print(bridge.shape)

#     # Decoder now
#     up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
#     print(up1.shape)
#     concat_1 = concatenate([up1, base_VGG.get_layer("block4_conv3").output], axis=3)
#     conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat_1)
#     conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

#     up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
#     print(up2.shape)
#     concat_2 = concatenate([up2, base_VGG.get_layer("block3_conv3").output], axis=3)
#     conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat_2)
#     conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

#     up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
#     print(up3.shape)
#     concat_3 = concatenate([up3, base_VGG.get_layer("block2_conv2").output], axis=3)
#     conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat_3)
#     conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

#     up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
#     print(up4.shape)
#     concat_4 = concatenate([up4, base_VGG.get_layer("block1_conv2").output], axis=3)
#     conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat_4)
#     conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

#     conv10 = Conv2D(NUM_CLASSES, 3, padding="same", activation="softmax")(conv9) #HO CAMBIATO QUESTO    
#     print(conv10.shape)


#     model_ = Model(inputs=inputs, outputs=[conv10], name="VGG16_U-Net") #[base_VGG.input]

#     return model_

