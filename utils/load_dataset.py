#!/usr/bin/env python3
import os
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2

from params import IMAGE_SIZE, train_dir, test_dir


def get_path_arrays(dir_path, type):
    if type == "input":
        """Returns a list of path arrays of input images"""
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
        """Returns a list of path arrays of mask images"""
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
    # random.Random(1337).shuffle(train_input_img_paths)
    # random.Random(1337).shuffle(train_mask_img_paths)
    # print("Number of input samples:", len(train_input_img_paths))
    # print("Number of mask samples:", len(train_mask_img_paths))

    # return load_data(IMAGE_SIZE, train_input_img_paths, train_mask_img_paths)
    (images,masks)= load_data_cv2(IMAGE_SIZE, train_input_img_paths, train_mask_img_paths)
        ###############################################
    #Encode labels... but multi dim array so need to flatten, encode and reshape
    # from sklearn.preprocessing import LabelEncoder
    # labelencoder = LabelEncoder()
    # n, h, w = masks.shape
    # masks_reshaped = masks.reshape(-1,1)
    # masks_reshaped_encoded = labelencoder.fit_transform(np.ravel(masks_reshaped))
    # masks_encoded_original_shape = masks_reshaped_encoded.reshape(n, h, w)
    # np.unique(masks_encoded_original_shape)
    # masks_input = np.expand_dims(masks_encoded_original_shape, axis=3)

    #################################################
    #train_images = np.expand_dims(train_images, axis=3)
    #train_images = normalize(train_images, axis=1)    print(type(train_images)) 

    ######################TEST
    # print(type(train_mask))#numpy.ndarray
    # print(train_images.shape) #(60, 224, 224, 3)
    # print(train_mask.shape) #(60, 224, 224, 1)
    
    masks_input = np.expand_dims(masks, axis=3)
    return images, masks_input

def get_test_dataset():
    """Returns tuple (input, target)"""
    # random.Random(1337).shuffle(test_input_img_paths)
    # random.Random(1337).shuffle(test_mask_img_paths)
    test_input_img_paths = get_path_arrays(test_dir, "input")
    test_mask_img_paths = get_path_arrays(
        test_dir, "mask")  # è ovviamente vuoto
    # print("Number of input samples:", len(test_input_img_paths))
    # print("Number of mask samples:", len(test_mask_img_paths))

    # return  load_data(IMAGE_SIZE, test_input_img_paths, test_mask_img_paths)
    (images,masks) =load_data_cv2(IMAGE_SIZE, test_input_img_paths, test_mask_img_paths)
        ###############################################
    # # Encode labels... but multi dim array so need to flatten, encode and reshape
    # from sklearn.preprocessing import LabelEncoder
    # labelencoder = LabelEncoder()
    # n, h, w = masks.shape
    # masks_reshaped = masks.reshape(-1,1)
    # masks_reshaped_encoded = labelencoder.fit_transform(masks_reshaped)
    # masks_encoded_original_shape = masks_reshaped_encoded.reshape(n, h, w)

    # np.unique(masks_encoded_original_shape)
    # masks_input = np.expand_dims(masks_encoded_original_shape, axis=3)

    #################################################
    # train_images = normalize(train_images, axis=1)
    
    masks_input = np.expand_dims(masks, axis=3)
    return images, masks_input


# print allinamento immagine | maschera
# for input_path, mask_path in zip(input_img_paths, mask_img_paths):
#     print(input_path, "|", mask_path)

# #explore dataset: plot images (image, mask) vertically
# img_in = mpimg.imread(train_input_img_paths[3])
# img_m = mpimg.imread(train_mask_img_paths[3])
# plt.subplot(2,1,1)
# plt.imshow(img_in)
# plt.subplot(2,1,2)
# plt.imshow(img_m)
# plt.show()


#TODO: inizializzare meglio test_input_img_paths e test_mask_img_paths
test_input_img_paths = get_path_arrays(test_dir, "input")
test_mask_img_paths = get_path_arrays(
        test_dir, "mask")  # è ovviamente vuoto


#https://github.com/bnsreenu/python_for_microscopists/blob/master/210_multiclass_Unet_using_VGG_resnet_inception.py
#Capture training image info as a list
#TODO:check by plotting images
def load_data_cv2(IMAGE_SIZE, input_img_paths, mask_img_paths):
    images =[]
    for img_path in input_img_paths:
        img = cv2.imread(img_path, 1)       
        img = cv2.resize(img, IMAGE_SIZE)
        images.append(img)
        
    #Convert list to array for machine learning processing        
    images = np.array(images)

    #Capture mask/label info as a list
    masks = [] 
    for mask_path in mask_img_paths:
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        masks.append(mask)
            
    #Convert list to array for machine learning processing          
    masks = np.array(masks)
    return images, masks