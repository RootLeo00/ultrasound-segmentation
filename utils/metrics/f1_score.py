
# TODO: calculate f1 score (precision and recall) for multiclass segmentation
# source: https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/metrics/_classification.py#L1001

# doc: https://www.baeldung.com/cs/multi-class-f1-score
from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf
from keras import backend as K
import segmentation_models as sm

def f1score(y_true, y_pred):
    """
    Return the F1_score of each label.
    Args:
        y_true: 1d array-like, or label indicator array / sparse matrix --> Ground truth (correct) target values
        y_pred: 1d array-like, or label indicator array / sparse matrix --> Estimated targets as returned by a classifier.
        label: array-like, default=None --> the label to return the IoU for
        average{'micro', 'macro', 'samples', 'weighted', 'binary'} or None, default='binary' 

                'binary':

                    Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
                'micro':

                    Calculate metrics globally by counting the total true positives, false negatives and false positives.
                'macro':

                    Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                'weighted':

                    Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
                'samples':

                    Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).

    Returns:
        the F1_score of each label
    """
    f1_score(y_true,
             y_pred, 
             labels=[0,1,2], #TODO:migliorare
             average=None,
             )
    
#https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras
### Define F1 measures: F1 = 2 * (precision * recall) / (precision + recall)
#Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
def custom_f1_macro(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    
def f1score_sm():
    sm.metrics.FScore(threshold=0.5)
    

