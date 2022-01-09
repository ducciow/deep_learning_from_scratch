import numpy as np
from .Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.prob = probability
        self.loc_zeros = None  # location matrix for preserving zeros

    def forward(self, input_tensor):
        output_tensor = input_tensor.copy()
        if not self.testing_phase:
            probs = np.random.random(output_tensor.shape)
            self.loc_zeros = np.ones_like(probs)
            self.loc_zeros[probs > self.prob] = 0  # set 0 with probability 1-p
            output_tensor *= self.loc_zeros
            output_tensor /= self.prob
        return output_tensor

    def backward(self, error_tensor):
        output_tensor = error_tensor.copy()
        output_tensor *= self.loc_zeros
        output_tensor /= self.prob
        return output_tensor
