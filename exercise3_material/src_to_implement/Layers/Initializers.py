import numpy as np


class Constant:
    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return np.ones(weights_shape) * self.value


class UniformRandom:
    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return np.random.uniform(0, 1, weights_shape)


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, sigma, weights_shape)


class He:
    def initialize(self, weights_shape, fan_in, fan_out=None):
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(0, sigma, weights_shape)