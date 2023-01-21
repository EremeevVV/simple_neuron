from src import non_linearity
import numpy as np


def test_sigmoid_fn():
    sigmoid = non_linearity.Sigmoid()
    given = np.array([1,0.5,-1])
    result = sigmoid._sigmoid_fn(given)
    assert result.sum() <= 1.62246 and len(result) == 3

def test_sigmoid_forward():
    sigmoid = non_linearity.Sigmoid()
    given = np.array([1,0.5,-1])
    sigmoid.forward(given)
    assert (sigmoid.X == given).all()


def test_sigmoid_backward():
    sigmoid = non_linearity.Sigmoid()
    given = np.array([1,0.5,-1])
    dz = np.array([0.5])
    sigmoid.X = given
    expected = np.array([0.09830597, 0.11750186, 0.09830597])
    dX = sigmoid.backward(dz)
    assert len(dX) == 3 and (dX.round(5) == expected.round(5)).all()


def test_relu_forward():
    given_x  = np.array([0,0.5,10,-4,-0.4])
    expected = np.array([0,0.5,10,0,0])
    relu = non_linearity.Relu()
    result = relu.forward(given_x)
    assert np.array_equal(result, expected) and np.array_equal(relu.X, given_x)


def test_relu_backward():
    relu = non_linearity.Relu()
    relu.X =  np.array([0,0.5,10,-4,-0.4])
    given_dz = np.array([5,5,5,5,5])
    expected = np.array([5,5,5,0,0])
    result = relu.backward(given_dz)
    assert np.array_equal(result,expected)