import numpy as np

from src.layer import Layer
from src.loss import LossFunction

from src.utils import compose


class Net:
    def __init__(self, layers: list[Layer], loss_fn: LossFunction):
        self.layers: list[Layer] = layers
        self.loss_fn: LossFunction = loss_fn

    def predict(self, x: np.array) -> np.array:
        forward_function = compose([layer.forward for layer in self.layers])
        return forward_function(x)

    def fit(self, x: np.array, y: np.array, num_iterations: int) -> np.array:
        if len(x) != len(y):
            raise ValueError(f'x size and y size must be equal')
        size = len(x)
        loss_val = 0
        backward_function = compose([layer.backward for layer in reversed(self.layers)])

        for i in range(num_iterations):
            for x_val, y_val in zip(x, y):
                y_hat = self.predict(x)
                loss_val += self.loss_fn.forward(y_hat, y_val)
                dz = self.loss_fn.backward()
                backward_function(dz)
            loss_val /= num_iterations
            print(f'Iteration {i} from {num_iterations} with {size=} have {loss_val=}.')
