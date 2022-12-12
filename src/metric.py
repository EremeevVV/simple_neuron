import numpy as np


def accuracy(y_pred: np.array, y_true: np.array) -> float:
    acc = np.sum(y_pred == y_true)
    return acc / y_true.shape[0]
