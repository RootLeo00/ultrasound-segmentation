#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras

from params import IMAGE_SIZE, model_dir, MODEL_NAME, NUM_CLASSES, BATCH_SIZE, EPOCHS
from utils.fine_tuning import finetune_unfreezeall
from utils.metrics import recall_m, precision_m
from utils.load_dataset import get_train_dataset

#TODO: inizializzare meglio
from models.unet import unet
model = unet(IMAGE_SIZE=IMAGE_SIZE, NUM_CLASSES=NUM_CLASSES)
history_TL=''
train_mask=''

def train():

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    start = time.time()
    (train_images, train_mask) = get_train_dataset()
    end = time.time()
    print("Seconds occurred to load train and test dataset: "+str(end-start))

    input_shape = IMAGE_SIZE + (3,)
    # from models.unet import unet
    # model = unet(IMAGE_SIZE=IMAGE_SIZE, NUM_CLASSES=NUM_CLASSES)
    # COMPILE MODEL

    if MODEL_NAME == "UNET":
        from models.unet import unet
        model = unet(IMAGE_SIZE=IMAGE_SIZE, NUM_CLASSES=NUM_CLASSES)
    elif MODEL_NAME == "TRANSFER_LEARNING_VGG16":
        from models.vgg16 import TL_unet_model
        model = TL_unet_model(input_shape=input_shape, NUM_CLASSES=NUM_CLASSES)
    elif MODEL_NAME == "TRANSFER_LEARNING_VGG19":
        from models.vgg19 import TL_unet_model
        model = TL_unet_model(input_shape=input_shape, NUM_CLASSES=NUM_CLASSES)
    """Config the model with losses and metrics with model.compile()"""
    print("shape of the mask: " + str(train_mask[0].shape))
    
    #IMBALANCED CLASSIFICATION - CLASS WEIGHTS
    """Class weights calculated by sklearn"""
    # class_weights = [np.ceil((100-percentages[0])/10), np.ceil((100-percentages[1])/10), np.ceil((100-percentages[2])/10)]
    from sklearn.utils import compute_class_weight
    class_weights = compute_class_weight(
                                                    'balanced',
                                                    classes=np.unique(train_mask.flatten()),
                                                    y=np.ravel(train_mask, order='C')
    )
    print(class_weights)
    # """My calculate class weights with percentage of pixels"""
    # # calculate weigths of weightedLoss (mi baso sulla priam immagine)
    # uniques, counts = np.unique(train_mask[0].flatten(), return_counts=True)
    # percentages = dict(zip(uniques, counts * 100 /
    #                    len(train_mask[0].flatten())))
    # print(percentages)
    # class_weights = {0: np.ceil((100-percentages[0])/10), 1: np.ceil((100-percentages[1])/10), 2: np.ceil((100-percentages[2])/10) }
    # print(class_weights)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, name="Adam")
    # loss=weightedLoss(tf.keras.losses.SparseCategoricalCrossentropy(), class_weights)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ["accuracy"]
    
    
    start = time.time()
    model.compile(
        # lr=learning_rate: we want to change it gently (provare però a metterlo più alto)
        optimizer=optimizer,
        # !!! We use the "sparse" version of categorical_crossentropy because our target data is integers.
        loss=loss,
        loss_weights=class_weights,
        metrics=metrics
    )
    end = time.time()
    print("Seconds occurred to compile model: "+str(end-start))

    checkpoint_path = model_dir+'/'+MODEL_NAME+'.hdf5'
    model_checkpoint = ModelCheckpoint(checkpoint_path,
                                       verbose=1,
                                       monitor='val_accuracy',
                                       save_best_only=True)
    cp_callback = model_checkpoint

    import datetime
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    print('\n\n--------------FITTING--------------------------')
    # Train the model, doing validation at the end of each epoch. (?)
    monitor = 'val_accuracy'
    # Stop training when a monitored metric has stopped improving.
    early_stopping = EarlyStopping(monitor=monitor, patience=20, verbose=1)
    """train the model with model.fit()"""
    callbacks = [tensorboard_callback, cp_callback, early_stopping]  # early_stopping
    start = time.time()
    history_TL = model.fit(x=train_images, 
                             y=train_mask,
                             batch_size=BATCH_SIZE,
                             epochs=EPOCHS,
                             verbose=2,
                             validation_split=0.2,  # NB The validation data is selected from the last samples in the x and y data provided, before shuffling.
                             callbacks=callbacks
                             )
    end = time.time()
    print("Seconds occurred to fit the model: "+str(end-start))

    #FINE TUNING
    if (MODEL_NAME == "TRANSFER_LEARNING_VGG16" or MODEL_NAME == "TRANSFER_LEARNING_VGG19"):
        # valutare se il fine tuning migliora il transfer learning base o no !!!!!!!!!!!!!!!!1
        # FINE TUNING: Unfreeze the contracting path and retrain it (devo ancora implementare KFold Cross validation)
        TL_checkpoint_path = model_dir+'/'+MODEL_NAME+'.hdf5'
        TLmodel = load_model(TL_checkpoint_path, custom_objects={'recall_m': recall_m, 'precision_m': precision_m})  # 'weightedLoss': weightedLoss,
        FTmodel = finetune_unfreezeall(IMAGE_SIZE, TLmodel)
        # FTmodel = finetune_unfreezedeepestl(input_shape, TLmodel)

        # for the changes to the model to take affect we need to recompile
        # the model, this time using optimizer with a *very* small learning rate
        print("[INFO] re-compiling model...")

        # compile
        """Config the model with losses and metrics with model.compile()"""
        start = time.time()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-7, name="Adam")  # the model, this time using optimizer with a *very* small learning rate
        FTmodel.compile(
            # lr=learning_rate: we want to change it gently (provare però a metterlo più alto)
            optimizer=optimizer,
            # !!! We use the "sparse" version of categorical_crossentropy because our target data is integers.
            loss=loss,
            loss_weights=class_weights,
            metrics=metrics
        )
        end = time.time()
        print("Seconds occurred to compile model: "+str(end-start))

        # train the model again, this time fine-tuning *both* the final set
        # of CONV layers along with our set of FC layers
        """train the model with model.fit()"""
        start = time.time()
        history_FT = FTmodel.fit(
            x=train_images,
            y=train_mask,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=2,
            validation_split=0.2,  # NB The validation data is selected from the last samples in the x and y data provided, before shuffling.
            callbacks=callbacks
        )
        end = time.time()
        print("Seconds occurred to fit the model: "+str(end-start))
