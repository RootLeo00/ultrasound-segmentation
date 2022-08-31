import os
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
import tensorflow as tf

from params import BATCH_SIZE, IMAGE_SIZE, train_dir, test_dir, SIZE_X, SIZE_Y


def get_path_arrays(dir_path, type):
    #TODO: check 'type' input
    """
    Returns a sorted list of paths file names (fnames) in a given directory (dir paths). 
    If the type is the string 'mask' or 'type', only the filenames that include that string are returned.
    
    Args:
        dir_path: absolute directory path
        type: string chosen between 'input' or 'mask'
    Returns:
        a sorted list of string paths
    """
    if type == "input":
        return sorted(
            [
                os.path.join(dir_path, fname)
                for fname in os.listdir(dir_path)
                if fname.endswith(".png")
                # con not fname... non funziona (?)
                if fname.find("mask") == -1
            ]
        )
    if type == "mask":
        return sorted(
            [
                os.path.join(dir_path, fname)
                for fname in os.listdir(dir_path)
                if fname.endswith(".png")
                if fname.find("mask") != -1
            ]
        )


def load_data(IMAGE_SIZE, input_img_paths, mask_img_paths):
    """Returns tuple (input, target)"""

    # (0) -genera-> int invece (0, ) -genera-> tupla ## ((0, 1, 2) + (3, )) --> (0, 1, 2, 3) ## la query genera tupla (32, 128, 128, 3)
    x = np.zeros((len(input_img_paths),) + IMAGE_SIZE + (3,), dtype="float32")
    for j, path in enumerate(input_img_paths):
        # LOADING --> tf.keras.preprocessing.image.load_img() Loads an image into PIL format.
        img = load_img(path, target_size=IMAGE_SIZE)
        # ATT! target_size in load_img() deve combaciare con la size che do quando definisco la tupla x
        x[j] = img

    # la query genera tupla (32, 128, 128, 1) --> il risultato infatti è una maschera
    y = np.zeros((len(mask_img_paths),) + IMAGE_SIZE + (1,), dtype="uint8")
    for j, path in enumerate(mask_img_paths):
        img = load_img(path, target_size=IMAGE_SIZE, color_mode="grayscale")
        # aggiungo un asse in posizione 2 probabilità), ottengo (160, 160, 1)--> Insert a new axis that will appear at the axis position in the expanded array shape
        y[j] = np.expand_dims(img, 2)
        # Ground truth labels are 0,255 --> Divide by 255 to make them 0, 1: (NORMALIZATION)
        # y[j] = y[j]/255.0
    return x, y


def get_train_dataset():
    """Returns tuple (input, target)"""
    train_input_img_paths = get_path_arrays(train_dir, "input")
    train_mask_img_paths = get_path_arrays(train_dir, "mask")

    (images,masks)= load_data_cv2(IMAGE_SIZE, train_input_img_paths, train_mask_img_paths)

    masks_input = np.expand_dims(masks, axis=3)
    return images, masks_input


def get_test_dataset():
    """Returns tuple (input, target)"""

    test_input_img_paths = get_path_arrays(test_dir, "input")
    test_mask_img_paths = get_path_arrays(test_dir, "mask")  # obv empty

    (images, masks) = load_data_cv2(
        IMAGE_SIZE, test_input_img_paths, test_mask_img_paths
    )
    masks_input = np.expand_dims(masks, axis=3)
    return images, masks_input

# TODO: inizializzare meglio test_input_img_paths e test_mask_img_paths
test_input_img_paths = get_path_arrays(test_dir, "input")
test_mask_img_paths = get_path_arrays(test_dir, "mask")  # è ovviamente vuoto


# https://github.com/bnsreenu/python_for_microscopists/blob/master/210_multiclass_Unet_using_VGG_resnet_inception.py
# Capture training image info as a list
# TODO:check by plotting images
def load_data_cv2(IMAGE_SIZE, input_img_paths, mask_img_paths):
    images = []
    for img_path in input_img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)

    # Convert list to array for machine learning processing
    images = np.array(images)

    # Capture mask/label info as a list
    masks = []
    for mask_path in mask_img_paths:
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(
            mask, (SIZE_Y, SIZE_X), #interpolation=cv2.INTER_NEAREST
        )  # Otherwise ground truth changes due to interpolation
        masks.append(mask)

    # Convert list to array for machine learning processing
    masks = np.array(masks)
    return images, masks

