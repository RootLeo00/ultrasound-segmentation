"""
    It is important to remember that in multi-class classification, we calculate the F1 score for each class in a One-vs-Rest (OvR) approach
    instead of a single overall F1 score, as seen in binary classification.
"""


def f1score_per_label(history, label):
    # get precision and recall from history
    precision = history["precision" + str(label)]
    recall = history["recall" + str(label)]
    _f1score = []
    for (p, r) in zip(precision, recall):
        _f1score.append(2 * ((p * r) / (p + r)))

    return _f1score

def val_f1score_per_label(history, label):
    # get precision and recall from history
    val_precision = history["val_precision" + str(label)]
    val_recall = history["val_recall" + str(label)]
    val_f1score = []
    for (p, r) in zip(val_precision, val_recall):
        val_f1score.append(2 * ((p * r) / (p + r)))

    return val_f1score


def f1score_weighted_average(history, class_weights):
    """The weighted-averaged F1 score is calculated by taking the mean of all per-class F1 scores while considering each class's support.
        F1_weighted= sum_from_0_to_n [(F1_i * W_i)]

    Args:
        history (_type_): _description_
        class_weights (_type_): proportion of each class's support relative to the sum of all support values.
                                where 'support' refers to the number of actual occurrences of the class in the dataset.

    Returns:
        _type_: _description_

    Source: https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f
    """
    # get number of classes
    num_classes = len(class_weights)
    f1score_weighted = 0.0
    val_f1score_weighted=0.0
    # get f1 score per label
    for i in range(0, num_classes):
        print(history["f1score" + str(i)])
        print(class_weights[i])
        f1score = history["f1score" + str(i)] * class_weights[i]
        f1score_weighted += f1score
        val_f1score = history["val_f1score" + str(i)] * class_weights[i]
        val_f1score_weighted += val_f1score

    return f1score_weighted, val_f1score_weighted
