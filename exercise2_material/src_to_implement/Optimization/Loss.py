import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.y_hat = None
        self.eps = None

    def forward(self, prediction_tensor, label_tensor):
        self.y_hat = prediction_tensor
        self.eps = np.finfo(self.y_hat.dtype).eps
        loss = - np.sum(label_tensor * np.log(self.y_hat + self.eps))
        return loss

    def backward(self, label_tensor):
        return - label_tensor / (self.y_hat + self.eps)
