import datetime
import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.utils import compute_class_weight
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from params import (BATCH_SIZE, EPOCHS, MODEL_NAME, NUM_CLASSES, SIZE_X, SIZE_Y,
                    model_dir, pred_dir, patience, monitor)
from utils.fine_tuning import finetune_unfreezeall
from utils.load_dataset import get_train_dataset
from utils.metrics.f1_score import f1score_per_label, val_f1score_per_label

# some basic initialization
if MODEL_NAME == "UNET":
    from models.unet import get_model
elif MODEL_NAME == "TRANSFER_LEARNING_VGG16":
    from models.vgg16 import get_model

elif MODEL_NAME == "TRANSFER_LEARNING_VGG19":
    from models.vgg19 import get_model
input_shape = (SIZE_X, SIZE_Y) + (3,) #(width, height, channels)
model = get_model(input_shape=input_shape, num_classes=NUM_CLASSES)

train_params = dict()


# FOR "TRAIN AND PREDICT" PROGRAM
def train():

    #get the dataset
    (train_images, train_masks) = get_train_dataset()

    #change train mask to categorical: explanation from minute 20 of https://www.youtube.com/watch?v=F365vQ8EndQ
    train_masks_cat = to_categorical(train_masks, num_classes=NUM_CLASSES)
    train_masks_cat = train_masks_cat.reshape(
        (train_masks.shape[0], train_masks.shape[1],
         train_masks.shape[2], NUM_CLASSES)
    )

    #calculate class weights because of imbalanced dataset
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_masks.flatten()),
        y=np.ravel(train_masks, order="C"),
    )

    #some train parameters and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, name="Adam")
    loss = "categorical_crossentropy"
    metrics = [
        "accuracy",
        tf.keras.metrics.IoU(
            num_classes=NUM_CLASSES,
            target_class_ids=[
                0
            ],
            name="binary_iou0",
        ),
        tf.keras.metrics.IoU(
            num_classes=NUM_CLASSES,
            target_class_ids=[1],
            name="binary_iou1",
        ),
        tf.keras.metrics.IoU(
            num_classes=NUM_CLASSES,
            target_class_ids=[2],
            name="binary_iou2",
        ),
        tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES, name="mean_iou"),
        tf.keras.metrics.Precision(name="precision0", class_id=0),
        tf.keras.metrics.Precision(name="precision1", class_id=1),
        tf.keras.metrics.Precision(name="precision2", class_id=2),
        tf.keras.metrics.Recall(name="recall0", class_id=0),
        tf.keras.metrics.Recall(name="recall1", class_id=1),
        tf.keras.metrics.Recall(name="recall2", class_id=2),
        tf.keras.metrics.AUC(
            curve='PR',
            summation_method='interpolation',
            name='auc',
            multi_label=False,
        )

    ]

    #compile model
    model.compile(
        optimizer=optimizer, loss=loss,  metrics=metrics, loss_weights=class_weights,
    )

    #some fitting callbacks
    checkpoint_path = model_dir + "/" + MODEL_NAME + ".hdf5"
    mode = "max"
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, monitor=monitor, mode=mode, save_best_only=True
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    early_stopping = EarlyStopping(
        monitor=monitor, mode=mode, patience=patience, verbose=1)
    callbacks = [tensorboard_callback, cp_callback, early_stopping]

    #fit model
    start = time.time()
    history_TL = model.fit(
        x=train_images,
        y=train_masks_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        validation_split=0.2,
        callbacks=callbacks,
    )
    stop = time.time()
    timeTL = stop - start
    print(f"Fit TL time: {timeTL}s")
    #update history
    history_dict = history_TL.history

    #fine tuning the model
    if (
        MODEL_NAME == "TRANSFER_LEARNING_VGG16"
        or MODEL_NAME == "TRANSFER_LEARNING_VGG19"
    ):
        TL_checkpoint_path = model_dir + "/" + MODEL_NAME + ".hdf5"
        TLmodel = load_model(TL_checkpoint_path)
        FTmodel = finetune_unfreezeall(TLmodel)

        # re-compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-7, name="Adam")
        FTmodel.compile(
            optimizer=optimizer, loss=loss, loss_weights=class_weights, metrics=metrics
        )

        start = time.time()
        history_FT = FTmodel.fit(
            x=train_images,
            y=train_masks_cat,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=2,
            validation_split=0.2,
            callbacks=callbacks,
        )
        stop = time.time()
        timeFT = stop - start
        train_params['timeFT'] = timeFT
        train_params['history_FT'] = history_FT.history

        # concatenate transfer learning history with fine tuning history (parameters should be the same)
        for keyTL, keyFT in zip(history_TL.history, history_FT.history):
            if (keyTL == keyFT):
                [history_dict[keyTL].append(elem)
                 for elem in history_FT.history[keyFT]]

    # post training analisys - f1 score metric
    for label in range(0, NUM_CLASSES):
        history_TL.history["f1score" + str(label)] = f1score_per_label(
            history_dict, label
        )
        history_TL.history["val_f1score" + str(label)] = val_f1score_per_label(
            history_dict, label
        )

    ##SAVE MODEL AND TRAIN FEATURES #########################
    # save keras.callbacks.History into json
    os.mkdir(pred_dir)
    json.dump(history_dict, open(pred_dir + "/history.json", "w"))

    # save model
    model.save(model_dir + "/" + MODEL_NAME + ".hdf5")

    # save a copy of the weights into the pred folder
    model.save(pred_dir + "/" + MODEL_NAME + ".hdf5")

    # save train params
    train_params['model'] = (model)
    train_params['optimizer'] = (optimizer)
    train_params['number_of_train_img'] = (len(train_images))
    train_params['loss'] = (loss)
    train_params['class_weights'] = (class_weights)
    train_params['metrics'] = (metrics)
    train_params['tensorboard_log_dir'] = (log_dir)
    train_params['callbacks'] = (callbacks)
    train_params['monitor'] = (monitor)
    train_params['history_TL'] = (history_TL.history)
    train_params['shape_train_imgs'] = (train_images.shape)
    train_params['shape_train_masks'] = (train_masks.shape)
    train_params['monitor'] = (monitor)
    train_params['timeTL'] = (timeTL)
    train_params['patience'] = (patience)

    #################################################

# FOR "ONLY PREDICTIONS" PROGRAM
def load_model_with_weights(weights_path):
    # Create model with weights from hdf5 file
    print(weights_path)
    model.load_weights(weights_path)
    # save a copy of the weights into the pred folder
    if (not os.path.isdir(pred_dir)):
        os.mkdir(pred_dir)
    model.save(pred_dir + "/" + MODEL_NAME + ".hdf5")
