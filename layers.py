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

    def __init__(self):
        ...

    def forward(self, X: np.array):
        ...

    def backward(self, dz: np.array, learning_rate=0.001):
        ...
