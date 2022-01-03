import numpy as np
from .Base import BaseLayer
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join('..')))
# from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self._X = None

    def forward(self, input_tensor):
        self._X = input_tensor
        return np.maximum(0, self._X)

    def backward(self, error_tensor):
        result_e = error_tensor
        result_e[self._X <= 0] = 0
        return result_e
