import numpy as np
from .Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        if len(stride_shape) == 1:
            self.stride_shape = (stride_shape[0], stride_shape[0])
        else:
            self.stride_shape = stride_shape
        if len(pooling_shape) == 1:
            self.pooling_shape = (pooling_shape[0], pooling_shape[0])
        else:
            self.pooling_shape = pooling_shape
        self.maxima = None  # maxima locations
        self._X = None

    def forward(self, input_tensor):
        self._X = input_tensor
        b, c, y, x = self._X.shape
        field_size = self.pooling_shape
        y_out = int(np.ceil((y - field_size[0] + 1) / self.stride_shape[0]))
        x_out = int(np.ceil((x - field_size[1] + 1) / self.stride_shape[1]))
        output = np.zeros((b, c, y_out, x_out))
        self.maxima = np.zeros_like(output)
        for batch in range(b):
            for channel in range(c):
                idx_y = 0
                for y_ in range(0, y, self.stride_shape[0]):
                    if y_ + field_size[0] > y:
                        break
                    idx_x = 0
                    for x_ in range(0, x, self.stride_shape[1]):
                        if x_ + field_size[1] > x:
                            break
                        field = self._X[batch, channel, y_: y_ + field_size[0], x_: x_ + field_size[1]]
                        max_idx_field = np.argmax(field)
                        max_idx_y = max_idx_field // field_size[1] + y_
                        max_idx_x = max_idx_field % field_size[1] + x_
                        max_value = self._X[batch, channel, max_idx_y, max_idx_x]
                        # mark maxima locations
                        max_idx_input = max_idx_y * x + max_idx_x
                        self.maxima[batch, channel, idx_y, idx_x] = max_idx_input
                        # set maxima values
                        output[batch, channel, idx_y, idx_x] = max_value
                        idx_x += 1
                    idx_y += 1
        return output

    def backward(self, error_tensor):
        b, c, y, x = error_tensor.shape
        result_e = np.zeros_like(self._X)
        for batch in range(b):
            for channel in range(c):
                for y_ in range(y):
                    for x_ in range(x):
                        max_idx = self.maxima[batch, channel, y_, x_]
                        max_idx_y = int(max_idx // self._X.shape[3])
                        max_idx_x = int(max_idx % self._X.shape[3])
                        result_e[batch, channel, max_idx_y, max_idx_x] += error_tensor[batch, channel, y_, x_]
        return result_e
