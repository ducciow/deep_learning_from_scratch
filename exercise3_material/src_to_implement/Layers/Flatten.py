import numpy as np
from .Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.shape
        return np.reshape(input_tensor, (self.shape[0], int(np.prod(self.shape[1:]))))

    def backward(self, error_tensor):
        return np.reshape(error_tensor, self.shape)
