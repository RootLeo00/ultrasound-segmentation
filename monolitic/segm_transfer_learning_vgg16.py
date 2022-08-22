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


if len(sys.argv) != 4:
    print ("""Arguemnts error \n
        Usage:  python3 segm_unet.py dataset_path number_of_epochs comment \n""")
    sys.exit(0)
    
description=sys.argv[3]

base_dir=sys.argv[1]
model_dir=base_dir+"model"
train_dir=base_dir+"train"
test_dir=base_dir+"test"
MODEL_NAME="TRANSFER_LEARNING_VGG16"
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3
BATCH_SIZE = 8
EPOCHS=int(sys.argv[2])

def get_path_arrays(dir_path, type):
    if type=="input":
        """Returns a list of path arrays of input images"""
        return sorted(
            [
                os.path.join(dir_path, fname)
                for fname in os.listdir(dir_path)
                    if fname.endswith(".png")
                    if fname.find("mask")==-1 #con not fname... non funziona (?)
            ]
        )
    if type=="mask":
        """Returns a list of path arrays of mask images"""
        return sorted(
                [
                    os.path.join(dir_path, fname)
                    for fname in os.listdir(dir_path)
                        if fname.endswith(".png")
                        if fname.find("mask")!=-1
                ]
        )

def load_data( IMAGE_SIZE, input_img_paths, mask_img_paths):
        """Returns tuple (input, target)"""
        
        x = np.zeros((len(input_img_paths),) + IMAGE_SIZE + (3,), dtype="float32") #(0) -genera-> int invece (0, ) -genera-> tupla ## ((0, 1, 2) + (3, )) --> (0, 1, 2, 3) ## la query genera tupla (32, 128, 128, 3)
        for j, path in enumerate(input_img_paths):
            img = load_img(path, target_size=IMAGE_SIZE) #LOADING --> tf.keras.preprocessing.image.load_img() Loads an image into PIL format.
            x[j] = img #ATT! target_size in load_img() deve combaciare con la size che do quando definisco la tupla x
            
        y = np.zeros((len(mask_img_paths),) + IMAGE_SIZE + (1,), dtype="uint8") ## la query genera tupla (32, 128, 128, 1) --> il risultato infatti è una maschera
        for j, path in enumerate(mask_img_paths):
            img = load_img(path, target_size=IMAGE_SIZE, color_mode="grayscale") 
            y[j] = np.expand_dims(img, 2) # aggiungo un asse in posizione 2 probabilità), ottengo (160, 160, 1)--> Insert a new axis that will appear at the axis position in the expanded array shape
            # Ground truth labels are 0,255 --> Divide by 255 to make them 0, 1: (NORMALIZATION)
            # y[j] = y[j]/255.0
        return x, y


# actual loading of training set and test set
train_input_img_paths = get_path_arrays(train_dir, "input")
train_mask_img_paths = get_path_arrays(train_dir,"mask")
# random.Random(1337).shuffle(train_input_img_paths)
# random.Random(1337).shuffle(train_mask_img_paths)

test_input_img_paths = get_path_arrays(test_dir, "input")
test_mask_img_paths = get_path_arrays(test_dir, "mask") #è ovviamente vuoto

print("Number of input samples:", len(train_input_img_paths))
print("Number of mask samples:", len(train_mask_img_paths))

#print allinamento immagine | maschera
# for input_path, mask_path in zip(input_img_paths, mask_img_paths):
#     print(input_path, "|", mask_path)

# #explore dataset: plot images (image, mask) vertically 
# img_in = mpimg.imread(train_input_img_paths[3])
# img_m = mpimg.imread(train_mask_img_paths[3])
# plt.subplot(2,1,1)
# plt.imshow(img_in)
# plt.subplot(2,1,2)
# plt.imshow(img_m)
# plt.show()

start =time.time()
(train_images, train_mask)  = load_data(  IMAGE_SIZE, train_input_img_paths, train_mask_img_paths)
(test_images, test_mask) = load_data( IMAGE_SIZE, test_input_img_paths, test_mask_img_paths)
end=time.time()
print("Seconds occurred to load train and test dataset: "+str(end-start))

# #explore dataset AFTER PREPROCESSING: plot images (image, mask) vertically 
# fig=plt.figure(figsize =(9.0,6.0))
# # plt.subplot(2,1,1)
# fig.add_subplot(2,1,1)
# plt.imshow(tf.keras.preprocessing.image.array_to_img(train_images[3]))
# plt.colorbar()
# plt.axis('off')
# # plt.subplot(2,1,2)
# fig.add_subplot(2,1,2)
# plt.imshow(tf.keras.preprocessing.image.array_to_img(train_mask[3])) #interpolation="nearest"
# plt.colorbar()
# plt.axis('off')
# plt.show()


#DEFINE THE MODEL
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam

# Unet pre-trained model with VGG16 (weights: imagenet) 
def TL_unet_model(input_shape, NUM_CLASSES):
    # input: input_shape (height, width, channels) 
    # return model

    #layer di preproce3ssing input
    inputs = keras.Input(shape=input_shape)
    preprocess_input = tf.keras.applications.vgg16.preprocess_input(inputs)
    input_shape = input_shape

    base_VGG = VGG16(
                    include_top = False, #tolgo i classifier/fully connecred layers (no dense and output layers)
                   weights = "imagenet", # Load weights pre-trained on ImageNet.
                   input_tensor=preprocess_input,
                   input_shape = input_shape
                   )

    # freezing all layers in VGG16 (non serve trainarle perchè tanto carico i pesi )
    for layer in base_VGG.layers: 
        layer.trainable = False

# Create new model on top
    # the bridge (exclude the last maxpooling layer in VGG16) 
    bridge = base_VGG.get_layer("block5_conv3").output
    print(bridge.shape)

    # Decoder now
    up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
    print(up1.shape)
    concat_1 = concatenate([up1, base_VGG.get_layer("block4_conv3").output], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat_1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    print(up2.shape)
    concat_2 = concatenate([up2, base_VGG.get_layer("block3_conv3").output], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat_2)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    print(up3.shape)
    concat_3 = concatenate([up3, base_VGG.get_layer("block2_conv2").output], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat_3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    print(up4.shape)
    concat_4 = concatenate([up4, base_VGG.get_layer("block1_conv2").output], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat_4)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(NUM_CLASSES, 3, padding="same", activation="softmax")(conv9) #HO CAMBIATO QUESTO    
    print(conv10.shape)


    model_ = Model(inputs=inputs, outputs=[conv10], name="VGG16_U-Net") #[base_VGG.input]

    return model_


# Function for fine-tuning the model (unfreeze all layers) 
def finetune_unfreezeall(input_shape, base_model):
    input_shape = input_shape
    base_model = base_model
    
    # unfreeze the contracting path and freeze the expanding path
    for layer in base_model.layers:
        if layer.name in ['block1_conv1', 'block1_conv2', 'block1_pool',
                          'block2_conv1', 'block2_conv2', 'block2_pool', 
                          'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool',
                          'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
                          'block5_conv1', 'block5_conv2', 'block5_conv3']:
            layer.trainable = True
            
    return base_model    

# Function for fine-tuning the model (unfreeze deepest layers) 
def finetune_unfreezedeepest(input_shape, base_model):
    input_shape = input_shape
    base_model = base_model
    
    # unfreeze the contracting path and freeze the expanding path
    for layer in base_model.layers:
        if layer.name in ['block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
                          'block5_conv1', 'block5_conv2', 'block5_conv3']:
            layer.trainable = True
            
    return base_model    


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
input_shape=IMAGE_SIZE + (3,)
model = TL_unet_model(input_shape, NUM_CLASSES)
# model.summary()  


#COMPILE MODEL

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #Input 'y' of 'Mul' Op has type float32 that does not match type uint8 of argument 'x'.
    y_true_f=tf.cast(y_pred_f,tf.float32) #convert from uint8 to float32
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# tensorflow/keras
def recall_m(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

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


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator
def weightedLoss(originalLossFunc, weightsList):

    @rename('weightedLoss')
    def lossFunc(true, pred):

        axis = -1 #if channels last 
        #axis=  1 #if channels first


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index    
        classSelectors = K.argmax(true) #, axis=axis
            #if your loss is sparse, use only true as classSelectors

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index   

        #input 'y' of 'Equal' Op has type int64 that does not match type int32 of argument 'x'.
        classSelectors=tf.cast(classSelectors,tf.int32) #convert from int64 to int32
        classSelectors = [K.equal(i, classSelectors) for i in range(len(weightsList))]

        #casting boolean to float for calculations  
        #each tensor in the list contains 1 where ground true class is equal to its index 
        #if you sum all these, you will get a tensor full of ones. 
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred) 
        loss = loss * weightMultiplier

        return loss
    return lossFunc

"""Config the model with losses and metrics with model.compile()"""
print("shape of the mask: "+ str(train_mask[0].shape))
#calculate weigths of weightedLoss (mi baso sulla priam immagine)
uniques, counts = np.unique(train_mask[0].flatten(), return_counts=True)
percentages = dict(zip(uniques, counts * 100 / len(train_mask[0].flatten())))
print(percentages)
# class_weights = {0: np.ceil((100-percentages[0])/10), 1: np.ceil((100-percentages[1])/10), 2: np.ceil((100-percentages[2])/10) }
# print(class_weights)
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, name="Adam")
# loss=weightedLoss(tf.keras.losses.SparseCategoricalCrossentropy(), class_weights)
loss=tf.keras.losses.SparseCategoricalCrossentropy()
metrics= ["accuracy"]
class_weights=[ np.ceil((100-percentages[0])/10), np.ceil((100-percentages[1])/10), np.ceil((100-percentages[2])/10) ]
start=time.time()
model.compile(
                optimizer=optimizer, #lr=learning_rate: we want to change it gently (provare però a metterlo più alto)
                loss=loss, #!!! We use the "sparse" version of categorical_crossentropy because our target data is integers.
                loss_weights=class_weights,
                metrics=metrics
            )
end=time.time()
print("Seconds occurred to compile model: "+str(end-start))


checkpoint_path = model_dir+'/vgg16_transfer_learning.hdf5'
model_checkpoint = ModelCheckpoint(checkpoint_path,  
                                    verbose=1,
                                    monitor='val_accuracy',
                                    save_best_only=True)
cp_callback=model_checkpoint

import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print('\n\n--------------FITTING--------------------------')
# Train the model, doing validation at the end of each epoch. (?)
from keras.callbacks import EarlyStopping
monitor='val_accuracy'
early_stopping = EarlyStopping(monitor=monitor, patience=20, verbose=1) #Stop training when a monitored metric has stopped improving.
"""train the model with model.fit()"""
callbacks=[tensorboard_callback, cp_callback, early_stopping ]#early_stopping
start=time.time()
history_TL=model.fit(  x=train_images, 
            y=train_mask, 
            batch_size=BATCH_SIZE,
            epochs=EPOCHS, 
            verbose=2,
            validation_split=0.2, #NB The validation data is selected from the last samples in the x and y data provided, before shuffling. 
            callbacks=callbacks
        )
end=time.time()
print("Seconds occurred to fit the model: "+str(end-start))


#valutare se il fine tuning migliora il transfer learning base o no !!!!!!!!!!!!!!!!1
#FINE TUNING: Unfreeze the contracting path and retrain it (devo ancora implementare KFold Cross validation)
TL_checkpoint_path = model_dir+'/vgg16_transfer_learning.hdf5'
TLmodel = load_model(TL_checkpoint_path, custom_objects={'recall_m': recall_m, 'precision_m': precision_m })#'weightedLoss': weightedLoss, 
FTmodel = finetune_unfreezeall(input_shape, TLmodel)
# FTmodel = finetune_unfreezedeepestl(input_shape, TLmodel)

# for the changes to the model to take affect we need to recompile
# the model, this time using optimizer with a *very* small learning rate
print("[INFO] re-compiling model...")

#compile
"""Config the model with losses and metrics with model.compile()"""
start=time.time()
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7,name="Adam") # the model, this time using optimizer with a *very* small learning rate
FTmodel.compile(
                optimizer=optimizer, #lr=learning_rate: we want to change it gently (provare però a metterlo più alto)
                loss=loss, #!!! We use the "sparse" version of categorical_crossentropy because our target data is integers.
                loss_weights=class_weights,
                metrics=metrics
            )
end=time.time()
print("Seconds occurred to compile model: "+str(end-start))

# train the model again, this time fine-tuning *both* the final set
# of CONV layers along with our set of FC layers
"""train the model with model.fit()"""
start=time.time()
history_FT = FTmodel.fit(
            x=train_images, 
            y=train_mask, 
            batch_size=BATCH_SIZE,
            epochs=EPOCHS, 
            verbose=2,
            validation_split=0.2, #NB The validation data is selected from the last samples in the x and y data provided, before shuffling. 
            callbacks=callbacks
)
end=time.time()
print("Seconds occurred to fit the model: "+str(end-start))


    #ANALISYS
pred_dir=base_dir+"predictions"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(pred_dir)
# Accuracy vs Epoch
def Accuracy_Graph(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    #plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.savefig(pred_dir+"/accuracy_graph.png")
    plt.show()
    
# Dice Similarity Coefficient vs Epoch
def Dice_coefficient_Graph(history):

    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    #plt.title('Dice_Coefficient')
    plt.ylabel('Dice_Coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.savefig(pred_dir+"/dice_coeff_graph.png")
    plt.show()

# Precision metric vs Epoch
def Precision_Graph(history):

    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    #plt.title('Dice_Coefficient')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.savefig(pred_dir+"/precision_graph.png")
    plt.show()

# Precision metric vs Epoch
def Recall_Graph(history):

    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    #plt.title('Dice_Coefficient')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.savefig(pred_dir+"/recall_graph.png")
    plt.show()


# Loss vs Epoch
def Loss_Graph(history):

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.savefig(pred_dir+"/loss_graph.png")
    plt.show()
Accuracy_Graph(history_TL)
Accuracy_Graph(history_FT)
Loss_Graph(history_TL)
Loss_Graph(history_FT)
# Precision_Graph(history_FT)
# Precision_Graph(history_TL)
# Recall_Graph(history_FT)
# Recall_Graph(history_TL)


#PREDICTION
print('\n\n--------------PREDICT--------------------------')
# Generate predictions for all images in the validation set
# (val_images, val_mask) = load_data(BATCH_SIZE, IMAGE_SIZE, val_input_img_paths, val_mask_img_paths)
"""use the model to do prediction with model.predict()"""
mask_predictions = model.predict(test_images,
                                verbose=2)


def display_mask(i, predictions):
    """Quick utility to display a model's prediction."""
    plt.subplot(3,1,1)
    test_image=load_img(test_input_img_paths[i])
    plt.imshow(test_image)

    plt.subplot(3,1,2)
    predicted_mask = np.argmax(predictions[i], axis=-1) #Returns the indices of the maximum values along an axis (nel mio caso, l'ultimo).
    predicted_mask = np.expand_dims(predicted_mask, axis=-1)
    predicted_mask=predicted_mask*255 #va moltiplicato per 255 perchè print(predicted_mask.tolist()) torna array con solo valori 0 e 1 
    w, h = test_image.size
    # print("height:"+str(h))
    # print("width:"+str(w))
    predicted_mask=tf.image.resize(predicted_mask, [h, w])
    predicted_mask = (tf.keras.preprocessing.image.array_to_img(predicted_mask)) #devo moltiplicare si o no per 255???? --> prova a st
    plt.imshow(predicted_mask)

    plt.subplot(3,1,3)
    plt.imshow(mpimg.imread(test_mask_img_paths[i]))
    plt.savefig(pred_dir+"/prediction_"+str(i)+".png")
    plt.show()

# Display results for prediction
for i in range(3):
    display_mask(i, mask_predictions)


#make output file prediction.txt
f = open(pred_dir+"/prediction"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".txt", "a")
f.write("--START--\n")
f.write("Using dataset at: "+base_dir+"\n")
f.write("IMAGE SIZE: "+str(IMAGE_SIZE)+"\n")
f.write("NUMBER OF CLASSES: "+str(NUM_CLASSES)+"\n")
f.write("BATCH SIZE: "+str(BATCH_SIZE)+"\n")
f.write("NUMBER OF EPOCHS: "+str(EPOCHS)+"\n")

f.write("****MY COMMENT****\n")
f.write(description)
f.write("\n*********\n")

f.write("Using "+str(len(train_input_img_paths))+" train images loaded from: "+train_dir+"\n")
f.write("Using "+str(len(test_input_img_paths))+" test images are loaded from: "+test_dir+"\n")
f.write("The model used for this training is: "+ MODEL_NAME+"\n")
f.write("Model summary:\n")
model.summary(print_fn=lambda x: f.write(x + '\n'))
f.write("\n--COMPILE AND EVALUATE MODEL--\n")
f.write("optimizer: "+str(optimizer)+"\n")
f.write("loss: "+ str(loss) +"\n")
f.write("with class_weights: "+str(class_weights)+"\n")
f.write("metrics: \n")
for metric in metrics :
    f.write(str(metric)+"\n")
f.write("Seconds occurred to compile model: "+str(end-start)+"\n")
f.write("using Tensorboard to evaluate training at log dir: "+log_dir+"\n")
f.write("\n--FITTING MODEL--\n")
f.write("callbacks:\n")
for callback in callbacks :
    f.write(str(callback)+"\n")
f.write("monitor metric: "+str(monitor)+"\n")
f.write("Transfer Learning history\n")
for key in history_TL.history.keys() :
    f.write(key+" : "+ str(history_TL.history[str(key)])+"\n")
f.write("Fine Tuning history\n")
for key in history_FT.history.keys() :
    f.write(key+" : "+ str(history_FT.history[str(key)])+"\n")
f.write("\n--PREDICTIONS MODEL--\n")
f.write("prediction images are saved at: "+pred_dir+"\n")
f.write("\n--END--")
f.close()