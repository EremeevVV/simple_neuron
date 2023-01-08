import numpy as np

from src.layer import Layer
from src.loss import LossFunction

from src.utils import compose


class Net:
    def __init__(self, layers: list[Layer], loss: LossFunction):
        self.layers = layers
        self.loss = loss

    def forward(self, X: np.array) -> np.array:
        forward_function = compose([layer.forward for layer in self.layers])
        return forward_function(X)

    def backward(self):
        ...
