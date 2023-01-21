from src.layer import Layer

import numpy as np


class Sigmoid(Layer):
    X: np.array

    def forward(self, X: np.array) -> np.array:
        self.X = X
        return self._sigmoid_fn(X)

    def backward(self, dz: np.array, learning_rate=0.001) -> np.array:
        return dz * self._sigmoid_fn(self.X) * (1 - self._sigmoid_fn(self.X))

    @staticmethod
    def _sigmoid_fn(X: np.array) -> np.array:
        return 1 / (1 + np.e ** (-1 * X))


class Relu(Layer):
    X: np.array

    def forward(self, X: np.array) -> np.array:
        self.X = X
        return np.maximum(0, X)

    def backward(self, dz: np.array) -> np.array:
        dz[self.X < 0] = 0
        return dz
