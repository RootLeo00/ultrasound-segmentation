import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D, Conv2DTranspose)
from tensorflow.keras.models import Model


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
def get_model(input_shape, num_classes):

    #layer di preproce3ssing input
    inputs = keras.Input(shape=input_shape) #input_shape=(widht, height, channels)
    preprocess_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    # load the VGG19 network, ensuring the head FC layer sets are left off
    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=preprocess_input, input_shape=input_shape)

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

    model = Model(vgg19.inputs, outputs, name="VGG19_U-Net")
    for layer in vgg19.layers:
        layer.trainable = False

    #compile in the train program to make it easy to test with various loss functions
    return model

