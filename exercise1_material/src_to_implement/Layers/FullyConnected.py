import numpy as np
from .Base import BaseLayer
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join('..')))
# from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, size=(input_size, output_size))
        self.bias = np.random.uniform(0, 1, size=output_size)
        self._optimizer = None
        self._X = None  # input tensor to current layer
        self._E = None  # error tensor to current layer
        self._weight_gradient = None  # gradient w.r.t. weights
        self._bias_gradient = None  # gradient w.r.t. bias

    def forward(self, input_tensor):
        self._X = input_tensor
        return np.matmul(self._X, self.weights) + self.bias

    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    optimizer = property(get_optimizer, set_optimizer)

    def backward(self, error_tensor):
        self._E = error_tensor
        # gradient wrt X
        result_e = np.matmul(self._E, self.weights.transpose())
        # gradient wrt W
        self._weight_gradient = np.matmul(self._X.transpose(), self._E)
        # gradient wrt bias
        self._bias_gradient = self._E
        # update weights
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._weight_gradient)
            self.bias = self._optimizer.calculate_update(self.bias, self._bias_gradient)
        return result_e

    @property
    def gradient_weights(self):
        return self._weight_gradient
