#!/usr/bin/env python3
import datetime
import json

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

from params import BATCH_SIZE, EPOCHS, IMAGE_SIZE, MODEL_NAME, NUM_CLASSES, model_dir
from utils.fine_tuning import finetune_unfreezeall
from utils.load_dataset import get_train_dataset
from utils.metrics.f1_score import f1score_per_label, val_f1score_per_label
from utils.metrics.iou import mean_iou
from sklearn.utils import compute_class_weight
from tensorflow.keras.models import load_model

# TODO: inizializzare meglio
if MODEL_NAME == "UNET":
    from models.unet import get_model
elif MODEL_NAME == "TRANSFER_LEARNING_VGG16":
    from models.vgg16 import get_model

    # get_model_sm(num_classes=NUM_CLASSES)
elif MODEL_NAME == "TRANSFER_LEARNING_VGG19":
    from models.vgg19 import get_model
model = get_model(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)


def train():
    keras.backend.clear_session()
    
    (train_images, train_masks) = get_train_dataset()
    # print("Class values in the dataset are ... ", np.unique(train_masks))  # 0 is the background/few unlabeled

    # minuto 20 di https://www.youtube.com/watch?v=F365vQ8EndQ
    train_masks_cat = to_categorical(train_masks, num_classes=NUM_CLASSES)
    train_masks_cat = train_masks_cat.reshape(
        (train_masks.shape[0], train_masks.shape[1], train_masks.shape[2], NUM_CLASSES)
    )

    # IMBALANCED CLASSIFICATION - CLASS WEIGHTS
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_masks.flatten()),
        y=np.ravel(train_masks, order="C"),
    )
    # """My calculate class weights with percentage of pixels"""
    # # calculate weigths of weightedLoss (mi baso sulla priam immagine)
    # uniques, counts = np.unique(train_masks[0].flatten(), return_counts=True)
    # percentages = dict(zip(uniques, counts * 100 /
    #                    len(train_masks[0].flatten())))
    # class_weights = {0: np.ceil((100-percentages[0])/10), 1: np.ceil((100-percentages[1])/10), 2: np.ceil((100-percentages[2])/10) }

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, name="Adam")
    loss = "categorical_crossentropy"
    # TODO: ciclo for per le metriche
    metrics = [
        "accuracy",
        tf.keras.metrics.IoU(
            num_classes=NUM_CLASSES,
            target_class_ids=[
                0
            ],  # If target_class_ids has only one id value, the IoU of that specific class is returned.
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
    ]

    model.compile(
        optimizer=optimizer, loss=loss, loss_weights=class_weights, metrics=metrics
    )

    checkpoint_path = model_dir + "/" + MODEL_NAME + ".hdf5"
    cp_callback = ModelCheckpoint(
        checkpoint_path, verbose=1, monitor="val_accuracy", save_best_only=True
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    print("\n\n--------------FITTING--------------------------")
    monitor = "val_accuracy"
    early_stopping = EarlyStopping(monitor=monitor, patience=20, verbose=1)
    callbacks = [tensorboard_callback, cp_callback, early_stopping]
    # import segmentation_models as sm
    # BACKBONE3 = 'vgg16'
    # preprocess_input3 = sm.get_preprocessing(BACKBONE3)

    # # preprocess input
    # X_train = preprocess_input3(train_images)
    history_TL = model.fit(
        x=train_images,
        y=train_masks_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        validation_split=0.2,
        callbacks=callbacks,
    )

    # FINE TUNING
    if (
        MODEL_NAME == "TRANSFER_LEARNING_VGG16"
        or MODEL_NAME == "TRANSFER_LEARNING_VGG19"
    ):
        TL_checkpoint_path = model_dir + "/" + MODEL_NAME + ".hdf5"
        TLmodel = load_model(TL_checkpoint_path, custom_objects={"mean_iou": mean_iou})
        FTmodel = finetune_unfreezeall(TLmodel)

        # compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-7, name="Adam")
        FTmodel.compile(
            optimizer=optimizer, loss=loss, loss_weights=class_weights, metrics=metrics
        )

        history_FT = FTmodel.fit(
            x=train_images,
            y=train_masks_cat,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=2,
            validation_split=0.2,
            callbacks=callbacks,
        )

    # TODO: concatenare le due history
    # POST TRAINING ANALISYS - F1 SCORE ANALISYS
    # Get the dictionary containing each metric and the loss for each epoch
    history_dict = history_TL.history
    for label in range(0, NUM_CLASSES):
        history_TL.history["f1score" + str(label)] = f1score_per_label(
            history_dict, label
        )
        history_TL.history["val_f1score" + str(label)] = val_f1score_per_label(
            history_dict, label
        )
    # save keras.callbacks.History  into json
    json.dump(history_dict, open(model_dir + "/history.json", "w"))

    # save model
    model.save(model_dir + "/vgg19_backbone_50epochs.hdf5")


def load_model_with_weights(weights_path):
    # Create model with weights from hdf5 file
    print(weights_path)
    model.load_weights(weights_path)
