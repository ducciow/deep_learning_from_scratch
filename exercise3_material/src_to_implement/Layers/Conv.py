import numpy as np
from scipy import signal
import copy
from .Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.num_kernels = num_kernels
        # 1-D weights: (out_channel, in_channel, height), stride: [int]
        if len(convolution_shape) == 2:
            self.convolution_shape = (self.num_kernels, *convolution_shape)
            self.stride_shape = stride_shape
        # 2-D weights: (out_channel, in_channel, height, width), stride: [int, int]
        else:
            self.convolution_shape = (self.num_kernels, *convolution_shape)
            if len(stride_shape) == 1:
                self.stride_shape = (stride_shape[0], stride_shape[0])
            else:
                self.stride_shape = stride_shape
        # init weights and bias
        self.fan_in = np.prod(convolution_shape)
        self.fan_out = self.num_kernels * np.prod(convolution_shape[1:])
        self.weights = np.random.uniform(0, 1, self.convolution_shape)
        self.bias = np.random.uniform(0, 1, self.num_kernels)
        self._gradient_weights = None
        self._gradient_bias = None
        self._X = None  # input tensor to current layer
        self._E = None  # error tensor to current layer
        # optimizers
        self._optimizer_weights = None
        self._optimizer_bias = None

    def forward(self, input_tensor):
        self._X = input_tensor
        # 1-D
        if len(self._X.shape) == 3:
            b, c, y = self._X.shape
            y_out = int(np.floor((y - 1) / self.stride_shape[0])) + 1
            c_out = self.num_kernels
            output = np.zeros((b, c_out, y_out))
            # cross correlation
            for batch in range(b):
                for kernel in range(c_out):
                    for channel in range(c):
                        input_arr = self._X[batch, channel]
                        filter_arr = self.weights[kernel, channel]
                        indices = [i for i in range(0, y, self.stride_shape[0])]
                        output[batch, kernel] += signal.correlate(input_arr, filter_arr, mode='same').take(indices)
                    # add bias
                    output[batch, kernel] += self.bias[kernel]
        # 2-D
        else:
            b, c, y, x = self._X.shape
            y_out = int(np.floor((y - 1) / self.stride_shape[0])) + 1
            x_out = int(np.floor((x - 1) / self.stride_shape[1])) + 1
            c_out = self.num_kernels
            output = np.zeros((b, c_out, y_out, x_out))
            # cross correlation
            for batch in range(b):
                for kernel in range(c_out):
                    for channel in range(c):
                        input_map = self._X[batch, channel]
                        filter_map = self.weights[kernel, channel]
                        indices = [i * x + j for i in range(0, y, self.stride_shape[0])
                                   for j in range(0, x, self.stride_shape[1])]
                        takeout = signal.correlate(input_map, filter_map, mode='same').take(indices) \
                            .reshape((y_out, x_out))
                        output[batch, kernel] += takeout
                    # add bias
                    output[batch, kernel] += self.bias[kernel]
        return output

    def backward(self, error_tensor):
        self._E = error_tensor
        # 1-D
        if len(self._E.shape) == 3:
            b, k, _ = self._E.shape
            c = self._X.shape[1]
            # gradients wrt. input
            result_e = np.zeros_like(self._X)
            weights_bw = np.zeros((c, k, self.convolution_shape[2]))
            for channel in range(c):
                for kernel in range(k):
                    weights_bw[channel, kernel] = self.weights[kernel, channel]
            for batch in range(b):
                for channel in range(c):
                    for kernel in range(k):
                        input_arr = self._E[batch, kernel]
                        input_arr = self._up_sample(input_arr)
                        filter_arr = weights_bw[channel, kernel]
                        result_e[batch, channel] += signal.convolve(input_arr, filter_arr, mode='same')
            # gradient wrt. weights
            self._gradient_weights = np.zeros(self.convolution_shape)
            for batch in range(b):
                for kernel in range(k):
                    for channel in range(c):
                        if self.convolution_shape[2] % 2 == 0:  # even kernel size
                            input_arr = np.zeros(self._X.shape[2] + self.convolution_shape[2] - 1)
                        else:
                            input_arr = np.zeros(self._X.shape[2] + 2 * int(np.floor(self.convolution_shape[2] / 2)))
                        fill_idx = int(np.floor(self.convolution_shape[2] / 2))
                        input_arr[fill_idx: fill_idx + self._X.shape[2]] = self._X[batch, channel]
                        filter_arr = self._E[batch, kernel]
                        filter_arr = self._up_sample(filter_arr)
                        self._gradient_weights[kernel, channel] += signal.correlate(input_arr, filter_arr,
                                                                                    mode='valid')
            self._gradient_weights = self._gradient_weights / b
            if self._optimizer_weights:
                self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
        # 2-D
        else:
            b, k, _, _ = self._E.shape
            c = self._X.shape[1]
            # gradients wrt. input
            result_e = np.zeros_like(self._X)
            weights_bw = np.zeros((c, k, self.convolution_shape[2], self.convolution_shape[3]))
            for channel in range(c):
                for kernel in range(k):
                    weights_bw[channel, kernel] = self.weights[kernel, channel]
            for batch in range(b):
                for channel in range(c):
                    for kernel in range(k):
                        input_arr = self._E[batch, kernel]
                        input_arr = self._up_sample(input_arr)
                        filter_arr = weights_bw[channel, kernel]
                        result_e[batch, channel] += signal.convolve(input_arr, filter_arr, mode='same')
            # gradient wrt. weights
            self._gradient_weights = np.zeros(self.convolution_shape)
            if self.convolution_shape[2] % 2 == 0:  # even kernel size in height
                padding_y = self.convolution_shape[2] - 1
            else:
                padding_y = 2 * int(np.floor(self.convolution_shape[2] / 2))
            if self.convolution_shape[3] % 2 == 0:  # even kernel size in width
                padding_x = self.convolution_shape[3] - 1
            else:
                padding_x = 2 * int(np.floor(self.convolution_shape[3] / 2))
            fill_idx_y = int(np.floor(self.convolution_shape[2] / 2))
            fill_idx_x = int(np.floor(self.convolution_shape[3] / 2))
            for batch in range(b):
                for kernel in range(k):
                    for channel in range(c):
                        input_arr = np.zeros((self._X.shape[2] + padding_y, self._X.shape[3] + padding_x))
                        input_arr[fill_idx_y: fill_idx_y + self._X.shape[2], fill_idx_x: fill_idx_x + self._X.shape[3]] \
                            = self._X[batch, channel]
                        filter_arr = self._E[batch, kernel]
                        filter_arr = self._up_sample(filter_arr)
                        self._gradient_weights[kernel, channel] += signal.correlate(input_arr, filter_arr,
                                                                                    mode='valid')
            # self._gradient_weights = self._gradient_weights / b
            if self._optimizer_weights:
                self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
        # gradient wrt. bias
        self._gradient_bias = np.zeros(self.num_kernels)
        for batch in range(b):
            for kernel in range(k):
                self._gradient_bias[kernel] += np.sum(self._E[batch, kernel])
        # self._gradient_bias = self._gradient_bias / b
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)
        return result_e

    def _up_sample(self, error_map):
        error_shape = error_map.shape
        x_shape = self._X.shape[2:]
        x_map = np.zeros(x_shape)
        # 1-D
        if len(error_shape) == 1:
            for i in range(error_shape[0]):
                j = self.stride_shape[0] * i
                x_map[j] = error_map[i]
        # 2-D
        else:
            for i_y in range(error_shape[0]):
                j_y = self.stride_shape[0] * i_y
                for i_x in range(error_shape[1]):
                    j_x = self.stride_shape[1] * i_x
                    x_map[j_y, j_x] = error_map[i_y, i_x]
        return x_map

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.convolution_shape, self.fan_in, self.fan_out)
        self.bias = bias_initializer.initialize(self.num_kernels, self.num_kernels, self.num_kernels)

    def norm(self):
        norm = 0
        if self._optimizer_weights.regularizer:
            norm += self._optimizer_weights.regularizer.norm(self.weights)
        if self._optimizer_bias.regularizer:
            norm += self._optimizer_bias.regularizer.norm(self.bias)
        return norm

    def get_optimizer(self):
        return self._optimizer_weights, self._optimizer_bias

    def set_optimizer(self, optimizer):
        self._optimizer_weights = copy.deepcopy(optimizer)
        self._optimizer_bias = copy.deepcopy(optimizer)

    optimizer = property(get_optimizer, set_optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias
