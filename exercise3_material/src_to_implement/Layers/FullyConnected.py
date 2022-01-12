import copy
import numpy as np
from .Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, size=(input_size, output_size))
        self.bias = np.random.uniform(0, 1, size=output_size)
        self._X = None  # input tensor to current layer
        self._E = None  # error tensor to current layer
        self._weight_gradient = None  # gradient w.r.t. weights
        self._bias_gradient = None  # gradient w.r.t. bias
        self._optimizer_weights = None
        self._optimizer_bias = None

    def forward(self, input_tensor):
        self._X = input_tensor.copy()
        return np.matmul(self._X, self.weights) + self.bias

    def get_optimizer(self):
        return self._optimizer_weights, self._optimizer_bias

    def set_optimizer(self, optimizer):
        self._optimizer_weights = copy.deepcopy(optimizer)
        self._optimizer_bias = copy.deepcopy(optimizer)

    optimizer = property(get_optimizer, set_optimizer)

    def backward(self, error_tensor):
        self._E = error_tensor.copy()
        # gradient wrt X
        result_e = np.matmul(self._E, self.weights.transpose())
        # gradient wrt W
        self._weight_gradient = np.matmul(self._X.transpose(), self._E)
        # gradient wrt bias
        self._bias_gradient = np.sum(self._E, axis=0)
        # update weights
        if self._optimizer_weights:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._weight_gradient)
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._bias_gradient)
        return result_e

    @property
    def gradient_weights(self):
        return self._weight_gradient

    def initialize(self, weights_initializer, bias_initializer):
        weights_shape = self.weights.shape
        bias_shape = self.bias.shape
        self.weights = weights_initializer.initialize(weights_shape, weights_shape[0], weights_shape[1])
        self.bias = bias_initializer.initialize(bias_shape, bias_shape[0], bias_shape[0])

    def norm(self):
        norm = 0
        if self._optimizer_weights.regularizer:
            norm += self._optimizer_weights.regularizer.norm(self.weights)
        if self._optimizer_bias.regularizer:
            norm += self._optimizer_bias.regularizer.norm(self.bias)
        return norm

    def get_input(self):
        return self._X

    def set_input(self, input_vector):
        self._X = input_vector.copy()

    curr_input = property(get_input, set_input)

