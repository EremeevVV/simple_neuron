import numpy as np

from src.loss import MSE


def test_forward():
    given_true = np.array([0.1, 0.9, 0.5])
    given_predict = np.array([0.2, 0.3, 0.6])
    loss = MSE().forward(given_predict, given_true)
    assert loss > 0


def test_backward():
    given_true = np.array([0.1, 0.9, 0.5])
    given_predict = np.array([0.2, 0.3, 0.6])
    expected = np.array([0.2, -1.2, 0.2])
    mse = MSE()
    mse.forward(given_predict, given_true)
    dz = mse.backward()
    print( dz == expected )
