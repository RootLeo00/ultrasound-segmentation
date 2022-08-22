#!/usr/bin/env python3
from calendar import EPOCH
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
import datetime
from tensorflow.keras import backend as K


if len(sys.argv) != 4:
    print ("""Arguemnts error \n
        Usage:  python3 segm_unet.py dataset_path number_of_epochs comment \n""")
    sys.exit(0)
    
description=sys.argv[3]
base_dir=sys.argv[1]
train_dir=base_dir+"train"
test_dir=base_dir+"test"
MODEL_NAME="UNET"
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
            
        y = np.zeros((len(mask_img_paths),) + IMAGE_SIZE + (1,), dtype="uint8") ## la query genera tupla (32, 128, 128, 1) --> il risultato infatti e' una maschera
        for j, path in enumerate(mask_img_paths):
            # img = tf.io.read_file(mask_img_paths)
            # img = tf.image.decode_png(img, channels=1)
            # img.set_shape([None, None, 1])
            # img = tf.image.resize(images=img, size=IMAGE_SIZE)
            # y=img
            img = load_img(path, target_size=IMAGE_SIZE, color_mode="grayscale") #grayscale
            y[j] = np.expand_dims(img, 2) # aggiungo un asse in posizione 2 probabilità), ottengo (160, 160, 1)--> Insert a new axis that will appear at the axis position in the expanded array shape
            # Ground truth labels are 0,255 --> Divide by 255 to make them 0, 1: (NORMALIZATION)
            # y[j] = y[j]/255.0
            #PER VEDERE CHE IL LOAD FUNZIONA SCOMMENTARE QUA SOTTO
            # if np.amax(y[j]>5):
            #     np.set_printoptions(threshold=sys.maxsize)
            #     print(np.hstack(y[j]))
            #     print(path)
            #     exit()
        return x, y


# actual loading of training set, valuation set and test set
# valuation_samples=80
train_input_img_paths = get_path_arrays(train_dir, "input")
train_mask_img_paths = get_path_arrays(train_dir,"mask")
# random.Random(1337).shuffle(train_input_img_paths)
# random.Random(1337).shuffle(train_mask_img_paths)

# train_input_img_paths = train_input_img_paths[:-valuation_samples]
# train_mask_img_paths = train_mask_img_paths[:-valuation_samples]

# val_input_img_paths = train_input_img_paths[-valuation_samples:]
# val_mask_img_paths = train_mask_img_paths[-valuation_samples:]

test_input_img_paths = get_path_arrays(test_dir, "input")
test_mask_img_paths = get_path_arrays(test_dir, "mask") #è ovviamente vuoto


print("Number of input samples:", len(train_input_img_paths))
print("Number of mask samples:", len(train_mask_img_paths))

#print allinamento immagine | maschera
# for input_path, mask_path in zip(input_img_paths, mask_img_paths):
#     print(input_path, "|", mask_path)

#explore dataset: plot images (image, mask) vertically 
img_in = mpimg.imread(train_input_img_paths[3])
img_m = mpimg.imread(train_mask_img_paths[3])
plt.subplot(2,1,1)
plt.imshow(img_in)
plt.subplot(2,1,2)
plt.imshow(img_m)
plt.show()

start =time.time()
(train_images, train_mask)  = load_data(  IMAGE_SIZE, train_input_img_paths, train_mask_img_paths)
# (val_images, val_mask) = load_data( IMAGE_SIZE, val_input_img_paths, val_mask_img_paths)
(test_images, test_mask) = load_data( IMAGE_SIZE, test_input_img_paths, test_mask_img_paths)
end=time.time()
print("Seconds occurred to load train and test dataset: "+str(end-start))

#explore dataset AFTER PREPROCESSING: plot images (image, mask) vertically 
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


def get_model(IMAGE_SIZE, NUM_CLASSES):
    inputs = keras.Input(shape=IMAGE_SIZE + (3,)) #aggiungo l'rgb (?)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same", input_shape=(128, 128, 3))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(NUM_CLASSES, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(IMAGE_SIZE, NUM_CLASSES)
# model.summary()

# Loads the weights
# checkpoint_path="/home/rootleo00/Downloads/unet_weights.hdf5"#weights.best.VGG19.hdf5
# model.load_weights(checkpoint_path)



def dice_coef(y_true, y_pred, smooth):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #Input 'y' of 'Mul' Op has type float32 that does not match type uint8 of argument 'x'.
    y_true_f=tf.cast(y_pred_f,tf.float32) #convert from uint8 to float32
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, M, smooth):
    y_true=tf.cast(y_pred,tf.float32) #convert from uint8 to float32
    dice = 0
    for index in range(M):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index], smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef_loss_multilabel(y_true, y_pred, M, smooth):
    return 1 - dice_coef_multilabel(y_true, y_pred, M, smooth)

# tensorflow/keras
def recall(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    # print(type(y_true))
    # print(type(y_pred))
    # print(y_true.shape)
    # print(y_pred.shape)
    # exit()
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


#COMPILE MODEL

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
        classSelectors = K.argmax(true, axis) #, axis=axis
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

def custom_dice_loss(y_actual,y_pred):
    custom_loss=dice_coef_loss_multilabel(y_actual, y_pred, M=NUM_CLASSES, smooth=0.1)
    return custom_loss
  
"""Config the model with losses and metrics with model.compile()"""
print("shape of the mask: "+ str(train_mask[0].shape))
#calculate weigths of weightedLoss (mi baso sulla priam immagine)
uniques, counts = np.unique(train_mask[0].flatten(), return_counts=True)
percentages = dict(zip(uniques, counts * 100 / len(train_mask[0].flatten())))
# print(percentages)
# class_weights = {0: np.ceil((100-percentages[0])/10), 1: np.ceil((100-percentages[1])/10), 2: np.ceil((100-percentages[2])/10) }
# print(class_weights)
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7, name="Adam")#tf.keras.optimizers.SGD(learning_rate=1e-7, name="SGD")
# loss=weightedLoss(keras.losses.sparse_categorical_crossentropy, class_weights)
loss=tf.keras.losses.SparseCategoricalCrossentropy()
metrics= ["accuracy", recall, precision]
class_weights=[ np.ceil((100-percentages[0])/10), np.ceil((100-percentages[1])/10), np.ceil((100-percentages[2])/10) ]#np.ceil((100-percentages[0])/10)
start=time.time()
model.compile(
                optimizer=optimizer, #sgd
                loss= loss,
                # loss_weights=class_weights,
                # loss=dice_coef_loss, #"sparse_categorical_crossentropy", #(mse) sparse_categorical_crossentropy!!! We use the "sparse" version of categorical_crossentropy because our target data is integers.
                # metrics=[dice_coef, 'accuracy', 'mse'] #tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()
                metrics= metrics
            )
end=time.time()
print("Seconds occurred to compile model: "+str(end-start))


#callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
monitor='val_accuracy'
earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=30, verbose=1)
# This callback will stop the training when there is no improvement in the loss for three consecutive epochs.



print('\n\n--------------FITTING--------------------------')
# Train the model, doing validation at the end of each epoch. (?)
"""train the model with model.fit()"""
callbacks=[tensorboard_callback, earlystopping_callback] #, earlystopping_callback
start=time.time()
history= model.fit(  x=train_images, 
            y=train_mask, 
            batch_size=BATCH_SIZE,
            epochs=EPOCHS, 
            # class_weight=class_weights,
            verbose=2,
            validation_split=0.2,
            callbacks=callbacks
        ) 
end=time.time()
print("Seconds occurred to fit model: "+str(end-start))

#EVALUATE THE MODEL --> lo faccio nel fitting
#evaluate accuracy
# val_loss, val_acc = model.evaluate(
#                                     x=val_images,
#                                     y=val_mask,
#                                     verbose=2)
# print('\nTest accuracy:', val_loss)
# print('Test loss:', val_acc)

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
Accuracy_Graph(history)
Loss_Graph(history)
Precision_Graph(history)
Recall_Graph(history)


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
for key in history.history.keys() :
    f.write(key+" : "+ str(history.history[str(key)])+"\n")
f.write("\n--PREDICTIONS MODEL--\n")
f.write("prediction images are saved at: "+pred_dir+"\n")
f.write("\n--END--")
f.close()
