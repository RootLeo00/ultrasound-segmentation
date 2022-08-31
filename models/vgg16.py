import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, concatenate

#DEFINE THE MODEL
def get_model(input_shape, num_classes):

    #layer di preprocessing input
    inputs = keras.Input(shape=input_shape)
    preprocess_input = tf.keras.applications.vgg16.preprocess_input(inputs)
    base_VGG = VGG16(
                    include_top = False, 
                   weights = "imagenet", # Load weights pre-trained on ImageNet.
                   input_tensor=preprocess_input,
                   input_shape = input_shape
                   )
    
    #freezing all layers in VGG16 (they already have imagenet weights)
    for layer in base_VGG.layers: 
        layer.trainable = False
    
    # Create new Decoder on top
    # Bridge (exclude the last maxpooling layer in VGG16) 
    bridge = base_VGG.get_layer("block5_conv3").output

    # Decoder
    up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
    concat_1 = concatenate([up1, base_VGG.get_layer("block4_conv3").output], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat_1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat_2 = concatenate([up2, base_VGG.get_layer("block3_conv3").output], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat_2)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    concat_3 = concatenate([up3, base_VGG.get_layer("block2_conv2").output], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat_3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    concat_4 = concatenate([up4, base_VGG.get_layer("block1_conv2").output], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat_4)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(num_classes, 3, padding="same", activation="softmax")(conv9) 


    model_ = Model(inputs=base_VGG.inputs, outputs=[conv10], name="VGG16_U-Net") #you can specify inputs as "base_VGG.inputs" as well as "inputs"
    
    return model_