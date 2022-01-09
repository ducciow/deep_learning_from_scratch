import numpy as np
from .Base import BaseLayer
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join('..')))
# from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self._X = None
        self.y_hat = None

    def forward(self, input_tensor):
        # shift x to increase numerical stability
        self._X = input_tensor
        self._X -= np.max(self._X, axis=1, keepdims=True)
        self.y_hat = np.exp(self._X) / np.sum(np.exp(self._X), axis=1, keepdims=True)
        return self.y_hat

    def backward(self, error_tensor):
        num_row = error_tensor.shape[0]
        num_column = error_tensor.shape[1]
        # calculate E_n-1 step-by-step according to slide 16
        weighted_sum = np.sum(self.y_hat * error_tensor, axis=1).repeat(num_column).reshape((num_row, num_column))
        result_e = self.y_hat * (error_tensor - weighted_sum)
        return result_e
