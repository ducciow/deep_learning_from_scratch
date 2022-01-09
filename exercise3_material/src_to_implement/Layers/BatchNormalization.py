import numpy as np
import copy
from .Base import BaseLayer
from .Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.eps = np.finfo(float).eps
        # weights and bias
        self.weights = None
        self.bias = None
        # initialize weights and bias
        self.initialize()
        # cache values
        self._X = None  # original x
        self._X_b = None  # normalized X before rescaling and shifting
        self._gamma_gradient = None
        self._beta_gradient = None
        self._optimizer_gamma = None
        self._optimizer_beta = None
        self.mu = None
        self.var = None
        self.avg_decay = 0.8
        self.mu_4_testing = None
        self.var_4_testing = None
        self.B = None
        self.M = None
        self.N = None
        self.it = 0

    def get_optimizer(self):
        return self._optimizer_gamma, self._optimizer_beta

    def set_optimizer(self, optimizer):
        self._optimizer_gamma = copy.deepcopy(optimizer)
        self._optimizer_beta = copy.deepcopy(optimizer)

    optimizer = property(get_optimizer, set_optimizer)

    def forward(self, input_tensor):
        self._X = input_tensor
        is_img = False
        # reformat
        if len(input_tensor.shape) > 2:
            is_img = True
            self._X = self.reformat(self._X)
        # testing phase
        if self.testing_phase:
            self._X_b = (self._X - self.mu_4_testing) / np.sqrt(self.var_4_testing + self.eps)
        else:
            self.mu = np.mean(self._X, axis=0)
            self.var = np.var(self._X, axis=0)
            # moving average for testing
            if self.it == 0:
                self.mu_4_testing = self.mu
                self.var_4_testing = self.var
            else:
                self.mu_4_testing = self.avg_decay * self.mu_4_testing + (1 - self.avg_decay) * self.mu
                self.var_4_testing = self.avg_decay * self.var_4_testing + (1 - self.avg_decay) * self.var
            self._X_b = (self._X - self.mu) / np.sqrt(self.var + self.eps)
        y_b = self.weights * self._X_b + self.bias
        if is_img:
            y_b = self.reformat(y_b)
        # increase iteration counter
        self. it += 1
        return y_b

    def backward(self, error_tensor):
        is_img = False
        # reformat
        if len(error_tensor.shape) > 2:
            is_img = True
            error_tensor = self.reformat(error_tensor)
        # gradient wrt weights
        self._gamma_gradient = np.sum(error_tensor * self._X_b, axis=0)
        # gradient wrt bias
        self._beta_gradient = np.sum(error_tensor, axis=0)
        # gradient wrt input
        result_e = compute_bn_gradients(error_tensor, self._X, self.weights, self.mu, self.var)
        # update weights
        if self._optimizer_gamma:
            self.weights = self._optimizer_gamma.calculate_update(self.weights, self._gamma_gradient)
        if self._optimizer_beta:
            self.bias = self._optimizer_beta.calculate_update(self.bias, self._beta_gradient)
        if is_img:
            result_e = self.reformat(result_e)
        return result_e

    def initialize(self):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def reformat(self, tensor):
        # image to vector
        if len(tensor.shape) > 2:
            B, H, M, N = tensor.shape
            tensor = tensor.reshape((B, H, M * N))
            tensor = np.transpose(tensor, (0, 2, 1))
            tensor = tensor.reshape((B * M * N, H))
            self.B = B
            self.M = M
            self.N = N
        # vector to image
        else:
            tensor = tensor.reshape((self.B, self.M * self.N, self.channels))
            tensor = np.transpose(tensor, (0, 2, 1))
            tensor = tensor.reshape((self.B, self.channels, self.M, self.N))
        return tensor

    @property
    def gradient_weights(self):
        return self._gamma_gradient

    @property
    def gradient_bias(self):
        return self._beta_gradient
