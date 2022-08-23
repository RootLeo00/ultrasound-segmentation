#!/usr/bin/env python3
import datetime
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from train import train, model

from main import base_dir
from utils.load_dataset import test_input_img_paths, test_mask_img_paths


# PREDICTION
def predict():
        
    pred_dir = base_dir+"predictions"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(pred_dir)
    (test_images, test_mask) = get_test_dataset()
    print('\n\n--------------PREDICT--------------------------')
    # Generate predictions for all images in the validation set
    # (val_images, val_mask) = load_data(BATCH_SIZE, IMAGE_SIZE, val_input_img_paths, val_mask_img_paths)
    # Display results for prediction
    """use the model to do prediction with model.predict()"""
    mask_predictions = model.predict(test_images,
                                    verbose=2)


    def display_mask(i, predictions):
        """Quick utility to display a model's prediction."""
        plt.subplot(3, 1, 1)
        test_image = load_img(test_input_img_paths[i])
        plt.imshow(test_image)

        plt.subplot(3, 1, 2)
        # Returns the indices of the maximum values along an axis (nel mio caso, l'ultimo).
        predicted_mask = np.argmax(predictions[i], axis=-1)
        predicted_mask = np.expand_dims(predicted_mask, axis=-1)
        # va moltiplicato per 255 perchè print(predicted_mask.tolist()) torna array con solo valori 0 e 1
        predicted_mask = predicted_mask*255
        w, h = test_image.size
        # print("height:"+str(h))
        # print("width:"+str(w))
        predicted_mask = tf.image.resize(predicted_mask, [h, w])
        # devo moltiplicare si o no per 255???? --> prova a st
        predicted_mask = (
            tf.keras.preprocessing.image.array_to_img(predicted_mask))
        plt.imshow(predicted_mask)

        plt.subplot(3, 1, 3)
        plt.imshow(mpimg.imread(test_mask_img_paths[i]))
        plt.savefig(pred_dir+"/prediction_"+str(i)+".png")
        plt.show()


    for i in range(3):
        display_mask(i, mask_predictions)