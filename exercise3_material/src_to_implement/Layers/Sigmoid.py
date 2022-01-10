import numpy as np
from .Base import BaseLayer


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.fxs = []

    def forward(self, input_tensor):
        fx = 1 / (1 + np.exp(- input_tensor))
        self.fxs.append(fx)
        return fx

    def backward(self, error_tensor):
        fx = self.fxs.pop()
        d_fx = fx * (1 - fx)
        return error_tensor * d_fx
