import json
import os
import tikzplotlib

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.metrics import multilabel_confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

from params import NUM_CLASSES, model_dir, pred_dir
from train import model
from utils.load_dataset import (get_test_dataset, test_input_img_paths,
                                test_mask_img_paths)
from utils.graph import graph

pred_params = dict()

def predict_func():

    ##GENERATE PREDICTIONS###############################################################
    if (not os.path.isdir(pred_dir)):
        os.mkdir(pred_dir)
    (test_images, test_masks) = get_test_dataset()
    print("\n\n--------------PREDICT--------------------------")
    pred_params['shape_test_imgs'] = (test_images.shape)
    pred_params['shape_test_masks'] = (test_masks.shape)

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
        predicted_mask = tf.keras.preprocessing.image.array_to_img(
            predicted_mask)
        plt.imshow(predicted_mask)

        plt.subplot(3, 1, 3)
        plt.imshow(mpimg.imread(fname=test_mask_img_paths[i]))
        plt.savefig(pred_dir + "/prediction_" + str(i) + ".png")
        # plt.show()
        plt.close()

    for i in range(3):
        display_mask(i, mask_predictions)

    ##EVALUATE PREDICTIONS #####################################################################
    history_dict = json.load(open(pred_dir + "/../history.json", "r")) #TODO: better path variables
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
    # Loss graph
    graph(history_dict,
          title='Loss Graph',
          xlabel='Epochs',
          ylabel='Loss',
          history_name='loss',
          history_val_name='val_loss',
          save_path=pred_dir+'/loss_graph.png')
    # AUC graph
    graph(history_dict,
          title='AUC Graph',
          xlabel='Epochs',
          ylabel='AUC',
          history_name='auc',
          history_val_name='val_auc',
          save_path=pred_dir+'/auc_graph.png')
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
    # Precision graph per label
    for label in range(0, NUM_CLASSES):
        graph(
            history_dict,
            title="Precision Graph For Label "+str(label),
            xlabel="Epochs",
            ylabel="Precision",
            history_name="precision"+str(label),
            history_val_name="val_precision"+str(label),
            save_path=pred_dir + "/precision_"+str(label)+"_graph.png",
        )
        # Recall graph per label
    for label in range(0, NUM_CLASSES):
        graph(
            history_dict,
            title="Recall Graph For Label "+str(label),
            xlabel="Epochs",
            ylabel="Recall",
            history_name="recall"+str(label),
            history_val_name="val_recall"+str(label),
            save_path=pred_dir + "/recall_"+str(label)+"_graph.png",
        )


    # plot classification report and multilabel confusion matrix
    from sklearn.metrics import classification_report
    y_true = test_masks
    y_pred = np.argmax(mask_predictions, axis=-1)
    report = classification_report(y_true.flatten(
        'C'), y_pred.flatten('C'),  output_dict=True)
    cm = multilabel_confusion_matrix(y_true.flatten(
        'C'), y_pred.flatten('C'), labels=[0, 1, 2])
    # leave transpose because csv file output gets transposed
    df = pandas.DataFrame(report).transpose()
    # print(df.T)
    df.to_csv(pred_dir+'/classification_report.csv')
    index = 0
    for cm_i in cm:
        df_confusion = pandas.DataFrame(cm_i).transpose()
        df_confusion.to_csv(pred_dir+'/confusion_matrix'+str(index)+'.csv')
        index = index+1

    # plot confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(
        y_true.flatten('C'), y_pred.flatten('C'))
    cm_savepath = pred_dir + "/multiclass_confusion_matrix.png"
    plt.savefig(cm_savepath)
    # save .tex pictures for latex 
    filename, file_extension = os.path.splitext(cm_savepath)
    tikzplotlib.save(cm_savepath.replace(file_extension, '.tex'))

