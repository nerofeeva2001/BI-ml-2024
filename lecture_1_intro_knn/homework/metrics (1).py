import numpy as np


def binary_classification_metrics(y_pred, y_true):
    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    false_positive = np.sum((y_pred == 1) & (y_true == 0))
    false_negative = np.sum((y_pred == 0) & (y_true == 1))
    true_negative = np.sum((y_pred == 0) & (y_true == 0))
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
    
    return precision, recall, f1, accuracy

def multiclass_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def r_squared(y_pred, y_true):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)



def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

    