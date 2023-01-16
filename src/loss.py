from abc import abstractmethod, ABC

import numpy as np
from numpy import ndarray


class LossFunction(ABC):
    @abstractmethod
    def forward(self, predict_val: np.array, true_val: np.array) -> np.array:
        ...

    @abstractmethod
    def backward(self) -> np.array:
        ...


class MSE(LossFunction):

    def __init__(self):
        self.predict_val = None
        self.true_val = None
        self.loss = None

    def forward(self, predict_val: np.array, true_val: np.array) -> ndarray:
        self.predict_val = predict_val
        self.true_val = true_val
        self.loss = np.sum((self.predict_val - self.true_val) ** 2)
        return self.loss

    def backward(self) -> np.array:
        return -2 * (self.true_val - self.predict_val)
