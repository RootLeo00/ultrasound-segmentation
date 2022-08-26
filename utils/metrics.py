#!/usr/bin/env python3
from keras.metrics import MeanIoU
import tensorflow as tf
# from IPython.display import Image, display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from params import pred_dir, NUM_CLASSES
import numpy as np

# TODO: copiare plot dei grafici da script (fa dei plot più carini)

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # Input 'y' of 'Mul' Op has type float32 that does not match type uint8 of argument 'x'.
    y_true_f = tf.cast(y_pred_f, tf.float32)  # convert from uint8 to float32
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


# IOU - Intersection Over Union
def iou(y_true, y_pred):
    y_pred_argmax = np.argmax(y_pred, axis=3)
    IOU_Keras = MeanIoU(num_classes=NUM_CLASSES)
    IOU_Keras.update_state(y_true[:, :, :, 0], y_pred_argmax)
    print("Mean IoU: "+ str(IOU_Keras.result().numpy()))
    iou_single_class(IOU_Keras)

# IOU single class - Intersection Over Union
def iou_single_class(IOU_Keras):
    values = np.array(IOU_Keras.get_weights()).reshape( NUM_CLASSES, NUM_CLASSES)
    denominatore = 0.0
    Class_IoU = [0 for i in range(0, NUM_CLASSES)] #TODO inizializza meglio
    for k in range(0, NUM_CLASSES):
        for i in range(0, NUM_CLASSES):
            for j in range(0, NUM_CLASSES):
                denominatore = denominatore+values[i, j]
        denominatore=denominatore+values[k, k]
        Class_IoU[k] = (values[k, k]/denominatore)
        print('Iou class '+ str(k)+ ' '+ str(Class_IoU[k]))


# GRAPHS
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
    # plt.title('Dice_Coefficient')
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
    # plt.title('Dice_Coefficient')
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
    # plt.title('Dice_Coefficient')
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

# IOU vs Epoch

def Iou_Graph(history):

    plt.figure()
    plt.title('Mean Iou graph')
    plt.ylabel('Iou')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.plot(history)
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig(pred_dir+"/iou_graph.png")
    plt.close()
