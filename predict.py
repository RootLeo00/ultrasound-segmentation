import json
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

from params import NUM_CLASSES, base_dir, model_dir, pred_dir
from train import model
from utils.load_dataset import (
    get_test_dataset,
    get_train_dataset,
    test_input_img_paths,
    test_mask_img_paths,
)
from utils.metrics.graph import graph


# PREDICTION
def predict_func():

    os.mkdir(pred_dir)
    (test_images, test_masks) = get_test_dataset()
    print("\n\n--------------PREDICT--------------------------")
    
    # Generate predictions for all images in the test set
    mask_predictions = model.predict(test_images, verbose=2)

    # Display results for prediction
    def display_mask(i, predictions):
        """Quick utility to display a model's prediction."""
        plt.subplot(3, 1, 1)
        test_image = load_img(test_input_img_paths[i])
        plt.imshow(test_image)

        plt.subplot(3, 1, 2)
        # consider the most probable class for each predicted pixel (model returns the probability of every class per pixel)
        predicted_mask = np.argmax(predictions[i], axis=-1)
        predicted_mask = np.expand_dims(predicted_mask, axis=-1)
        # convert to rgb
        predicted_mask = predicted_mask * 255
        w, h = test_image.size
        # resize to original shape
        predicted_mask = tf.image.resize(predicted_mask, [h, w])
        predicted_mask = tf.keras.preprocessing.image.array_to_img(predicted_mask)
        plt.imshow(predicted_mask)

        plt.subplot(3, 1, 3)
        plt.imshow(mpimg.imread(test_mask_img_paths[i]))
        plt.savefig(pred_dir + "/prediction_" + str(i) + ".png")
        # plt.show()

    for i in range(3):
        display_mask(i, mask_predictions)


##EVALUATE PREDICTIONS #####################################################################
    history_dict = json.load(open(model_dir + "/history.json", "r"))
    # Mean IOU graph
    graph(history_dict,
          title='Mean IoU Graph',
          xlabel='Epochs',
          ylabel='Mean IoU',
          history_name='mean_iou',
          history_val_name='val_mean_iou',
          save_path=pred_dir+'/mean_iou_graph.png')
    # Accuracy graph
    graph(history_dict,
          title='Accuracy Graph',
          xlabel='Epochs',
          ylabel='Accuracy',
          history_name='accuracy',
          history_val_name='val_accuracy',
          save_path=pred_dir+'/accuracy_graph.png')
    # F1 Score graph per label
    for label in range(0, NUM_CLASSES):
        graph(
            history_dict,
            title="F1 Score Graph For Label "+str(label),
            xlabel="Epochs",
            ylabel="F1 Score",
            history_name="f1score"+str(label),
            history_val_name="val_f1score"+str(label),
            save_path=pred_dir + "/f1_score_"+str(label)+"_graph.png",
        )

    
    ##WEIGHTED AVERAGE F1 SCORE################################################################


    #TODO: precision and recall for predictions
    #TODO: f1 for predictions
    #TODO: f1 average for predictions
    from utils.metrics.f1_score import f1score_weighted_average
    from tensorflow.keras.utils import to_categorical
    from sklearn.utils import compute_class_weight
        # minuto 20 di https://www.youtube.com/watch?v=F365vQ8EndQ
    test_masks_cat = to_categorical(test_masks, num_classes=NUM_CLASSES)
    test_masks_cat = test_masks_cat.reshape(
        (test_masks.shape[0], test_masks.shape[1], test_masks.shape[2], NUM_CLASSES)
    )

    # IMBALANCED CLASSIFICATION - CLASS WEIGHTS
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(test_masks.flatten()),
        y=np.ravel(test_masks, order="C"),
    )
    f1score_weighted_average(history_dict, class_weights=class_weights)


