import numpy as np


class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self, weights):
        return self.alpha * np.sum(weights ** 2)


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        gradients = np.zeros_like(weights)
        gradients[weights > 0] = 1
        gradients[weights < 0] = -1
        return self.alpha * gradients

    def norm(self, weights):
        return self.alpha * np.sum(np.abs(weights))
