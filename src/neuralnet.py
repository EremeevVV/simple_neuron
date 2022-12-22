from src.layer import Layer
from src.loss import LossFunction


class Net:
    def __init__(self, layers: list[Layer], loss: LossFunction):
        self.layers = layers
        self.loss = loss

    def froward(self):
        ...

    def backward(self):
        ...
