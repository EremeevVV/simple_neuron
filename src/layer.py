from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):

    @abstractmethod
    def forward(self, X: np.array):
        ...

    @abstractmethod
    def backward(self, dz: np.array, learning_rate=0.001):
        ...


class Dense(Layer):
    """Dense layer without batching"""

    def __init__(self, in_size: int, out_size: int, reg_lambda: float = 0.0):
        self.X = None
        self.W = np.random.normal(scale=1, size=(out_size, in_size)) * np.sqrt(
            2 / (in_size + out_size))  # инициализация весов
        self.B = np.zeros(out_size)
        self.reg_lambda = reg_lambda

    def forward(self, X: np.array) -> np.array:
        self.X = X
        if self._check_dimension(X):
            raise ValueError('X is not the same dimension as in_size')
        return np.dot(self.W, self.X) + self.B

    def backward(self, dz: np.array, learning_rate=0.001) -> np.array:
        dW = np.outer(self.X, dz)
        dB = dz
        dX = np.dot(dz, self.W)
        self.W -= learning_rate * dW
        self.B -= learning_rate * dB
        return dX

    def _check_dimension(self, X: np.array) -> bool:
        return X.shape[0] != self.W.shape[1]


