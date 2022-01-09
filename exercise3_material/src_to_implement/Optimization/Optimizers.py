import numpy as np


class Optimizer:
    def __init__(self, reg=None):
        self.regularizer = reg

    def add_regularizer(self, reg):
        self.regularizer = reg


class Sgd(Optimizer):
    def __init__(self, lr=0.001):
        super().__init__()
        self.lr = lr

    def calculate_update(self, weight_tensor, gradient_tensor):
        result = weight_tensor - self.lr * gradient_tensor
        if self.regularizer:
            return result - self.lr * self.regularizer.calculate_gradient(weight_tensor)
        else:
            return result


class SgdWithMomentum(Optimizer):
    def __init__(self, lr=0.001, mr=0.9):
        super().__init__()
        self.lr = lr
        self.mr = mr
        self.velocity = None
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.k < 1:
            self.velocity = np.zeros_like(weight_tensor)
        self.k += 1
        self.velocity = self.mr * self.velocity - self.lr * gradient_tensor
        result = weight_tensor + self.velocity
        if self.regularizer:
            return result - self.lr * self.regularizer.calculate_gradient(weight_tensor)
        else:
            return result


class Adam(Optimizer):
    def __init__(self, lr=0.001, mu=0.9, rho=0.999, epsilon=1e-8):
        super().__init__()
        self.lr = lr
        self.mu = mu
        self.rho = rho
        self.epsilon = epsilon
        self.v = None
        self.r = None
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.k < 1:
            self.v = np.zeros_like(weight_tensor)
            self.r = np.zeros_like(weight_tensor)
        self.k += 1
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * (gradient_tensor ** 2)
        v_corrected = self.v / (1 - self.mu ** self.k)
        r_corrected = self.r / (1 - self.rho ** self.k)
        result = weight_tensor - self.lr * v_corrected / (np.sqrt(r_corrected) + self.epsilon)
        if self.regularizer:
            return result - self.lr * self.regularizer.calculate_gradient(weight_tensor)
        else:
            return result
