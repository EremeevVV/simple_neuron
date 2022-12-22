from copy import copy

import numpy as np

from src.layer import Dense


def test_is_wrong_dimension_neg():
    layer = Dense(in_size=5, out_size=32)
    given = np.array([1,2,3,4,5])
    assert not layer._is_wrong_dimension(given)


def test_is_wrong_dimension_pos():
    layer = Dense(in_size=5, out_size=32)
    given = np.array([1,2,3,4,5,6])
    assert layer._is_wrong_dimension(given)


def test_forward():
    out_dim = 32
    layer = Dense(in_size=5, out_size=out_dim)
    given = np.array([1,2,3,4,5])
    result = layer.forward(given)
    assert np.array_equal(layer.X, given) and result.shape[0] == out_dim


def test_backward():
    in_dim = 5
    out_dim = 32
    layer = Dense(in_size=in_dim, out_size=out_dim)
    layer.X = np.array([1,2,3,4,5])
    unexpected = copy(layer.W)
    given_dz = np.array(out_dim*[0.5])
    # When
    result_dx = layer.backward(given_dz)
    assert not np.array_equal(unexpected,layer.W) and result_dx.shape[0]==in_dim