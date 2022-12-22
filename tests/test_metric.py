import numpy as np

from src.metric import accuracy


def test_accuracy():
    y_true = np.array([0., 1., 0., 1.])
    y_pred = np.array([0., 0., 1., 1.])
    expected = 0.5
    result = accuracy(y_pred, y_true)
    assert expected == result
