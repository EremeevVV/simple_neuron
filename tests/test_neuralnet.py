import numpy as np

from src.neuralnet import Net
from src.layer import Dense
from src.non_linearity import Sigmoid
from src.loss import MSE


def test_net_forward():
    in_dim = 7
    inner_dim = 16
    X_array = np.array([i / 10 for i in range(in_dim)])
    dense1 = Dense(in_size=in_dim, out_size=inner_dim)
    sigm1 = Sigmoid()
    dense2 = Dense(in_size=inner_dim, out_size=1)
    sigm2 = Sigmoid()
    loss = MSE()
    net = Net(layers=[dense1, sigm1, dense2, sigm2], loss=loss)
    # When
    result = net.forward(X_array)

    assert 0 < result[0] < 1 and len(result) == 1

def test_net_backward():
    in_dim = 7
    inner_dim = 16
    X_array = np.array([i / 10 for i in range(in_dim)])
    dense1 = Dense(in_size=in_dim, out_size=inner_dim)
    sigm1 = Sigmoid()
    dense2 = Dense(in_size=inner_dim, out_size=1)
    sigm2 = Sigmoid()
    loss = MSE()
    net = Net(layers=[dense1, sigm1, dense2, sigm2], loss=loss)
    # When
    result = net.forward(X_array)

    assert 0 < result[0] < 1 and len(result) == 1
