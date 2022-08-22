#!/usr/bin/env python3
import tensorflow as tf
# from IPython.display import Image, display
import tensorflow as tf
from tensorflow.keras import backend as K


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator
def weightedLoss(originalLossFunc, weightsList):

    @rename('weightedLoss')
    def lossFunc(true, pred):

        axis = -1 #if channels last 
        #axis=  1 #if channels first


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index    
        classSelectors = K.argmax(true) #, axis=axis
            #if your loss is sparse, use only true as classSelectors

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index   

        #input 'y' of 'Equal' Op has type int64 that does not match type int32 of argument 'x'.
        classSelectors=tf.cast(classSelectors,tf.int32) #convert from int64 to int32
        classSelectors = [K.equal(i, classSelectors) for i in range(len(weightsList))]

        #casting boolean to float for calculations  
        #each tensor in the list contains 1 where ground true class is equal to its index 
        #if you sum all these, you will get a tensor full of ones. 
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred) 
        loss = loss * weightMultiplier

        return loss
    return lossFunc
