import json
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

from params import NUM_CLASSES, model_dir, pred_dir
from train import model
from utils.load_dataset import (
    get_test_dataset,
    test_input_img_paths,
    test_mask_img_paths,
)


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
        plt.imshow(mpimg.imread(fname=test_mask_img_paths[i]))
        plt.savefig(pred_dir + "/prediction_" + str(i) + ".png")
        # plt.show()

    for i in range(3):
        display_mask(i, mask_predictions)


    ##EVALUATE PREDICTIONS #####################################################################
    history_dict = json.load(open(model_dir + "/history.json", "r"))
    # Mean IOU graph
    # graph(history_dict,
    #       title='Mean IoU Graph',
    #       xlabel='Epochs',
    #       ylabel='Mean IoU',
    #       history_name='mean_iou',
    #       history_val_name='val_mean_iou',
    #       save_path=pred_dir+'/mean_iou_graph.png')
    # # Accuracy graph
    # graph(history_dict,
    #       title='Accuracy Graph',
    #       xlabel='Epochs',
    #       ylabel='Accuracy',
    #       history_name='accuracy',
    #       history_val_name='val_accuracy',
    #       save_path=pred_dir+'/accuracy_graph.png')
    # # F1 Score graph per label
    # for label in range(0, NUM_CLASSES):
    #     graph(
    #         history_dict,
    #         title="F1 Score Graph For Label "+str(label),
    #         xlabel="Epochs",
    #         ylabel="F1 Score",
    #         history_name="f1score"+str(label),
    #         history_val_name="val_f1score"+str(label),
    #         save_path=pred_dir + "/f1_score_"+str(label)+"_graph.png",
    #     )

    ##CLASSIFICATION REPORT###############################################################
    from sklearn.metrics import classification_report
    target_names = [('class '+str(i)) for i in range(0,NUM_CLASSES)] #TODO: mettere nomi alle classi
    y_true=test_masks
    y_pred=np.argmax(mask_predictions, axis=-1)
    # print(multilabel_confusion_matrix(y_true.flatten('C'),y_pred.flatten('C'), labels=[0,1,2])) #TODO: array labels più carino
    print(classification_report(y_true.flatten('C'),y_pred.flatten('C')))


