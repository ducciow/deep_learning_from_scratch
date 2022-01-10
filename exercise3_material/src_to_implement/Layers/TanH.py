import numpy as np
from .Base import BaseLayer


class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.fxs = []

    def forward(self, input_tensor):
        fx = np.tanh(input_tensor)
        self.fxs.append(fx)
        return fx

    def backward(self, error_tensor):
        fx = self.fxs.pop()
        d_fx = 1 - fx ** 2
        return error_tensor * d_fx
