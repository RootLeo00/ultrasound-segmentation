import os
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2

from params import IMAGE_SIZE, train_dir, test_dir, SIZE_X, SIZE_Y

dataset_params=dict()

def get_path_arrays(dir_path, type):
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
    x = np.zeros((len(input_img_paths),) + IMAGE_SIZE + (3,), dtype="float32")
    for j, path in enumerate(input_img_paths):
        img = load_img(path, target_size=IMAGE_SIZE)
        x[j] = img

    y = np.zeros((len(mask_img_paths),) + IMAGE_SIZE + (1,), dtype="uint8")
    for j, path in enumerate(mask_img_paths):
        img = load_img(path, target_size=IMAGE_SIZE, color_mode="grayscale")
        y[j] = np.expand_dims(img, 2)
    return x, y

#initialization
test_input_img_paths = get_path_arrays(test_dir, "input")
test_mask_img_paths = get_path_arrays(test_dir, "mask")  # obv empty
train_input_img_paths = get_path_arrays(train_dir, "input")
train_mask_img_paths = get_path_arrays(train_dir, "mask")
dataset_params['num_train_images']=(len(train_input_img_paths))
dataset_params['num_test_images']=(len(test_input_img_paths))


def get_train_dataset():
    """Returns tuple (input, target)"""
    (images,masks)= load_data_cv2(IMAGE_SIZE, train_input_img_paths, train_mask_img_paths)
    #watch https://www.youtube.com/watch?v=vgdFovAZUzM&t=3s
    masks_input = np.expand_dims(masks, axis=3)
    return images, masks_input


def get_test_dataset():
    """Returns tuple (input, target)"""
    (images, masks) = load_data_cv2(
        IMAGE_SIZE, test_input_img_paths, test_mask_img_paths
    )
    # masks_input = np.expand_dims(masks, axis=3)
    return images, masks


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
            mask, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST
        )  # Otherwise ground truth changes due to interpolation
        masks.append(mask)

    # Convert list to array for machine learning processing
    masks = np.array(masks)
    return images, masks

