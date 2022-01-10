import numpy as np
import copy
from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # cell attributes
        self._memorize = False  # whether restore the hidden state from last iteration
        self.h = np.zeros(self.hidden_size)  # h_t
        self.y = np.empty(self.output_size)  # y_t
        self.xs = []  # length: time steps
        self.hs = []  # length: time steps + 1
        self.ys = []  # length: time steps
        self.h_fc = None
        self.y_fc = None
        self.tanh = None
        self.sigmoid = None
        # initialize component layers
        self.initialize_components()

    def forward(self, input_tensor):
        if not self._memorize:
            self.h = np.zeros(self.hidden_size)
            self.y = np.empty(self.output_size)
        self.xs = []
        self.hs = []
        self.hs.append(self.h)
        self.ys = []
        # iterate through time
        for x in input_tensor:
            self.xs.append(x)
            # -----> 1. compute h_t <-----
            # concatenate x_t and h_t-1
            h_n_x = np.concatenate((x, self.h))
            # add one dimension for batch
            h_n_x = np.expand_dims(h_n_x, axis=0)
            self.h = self.h_fc.forward(h_n_x)
            # reduce the first dimension
            self.h = np.squeeze(self.h)
            self.h = self.tanh.forward(self.h)
            self.hs.append(self.h)
            # compute y_t
            # add dimension
            self.h = np.expand_dims(self.h, axis=0)
            self.y = self.y_fc.forward(self.h)
            # reduce the first dimension
            self.h = np.squeeze(self.h)
            self.y = np.squeeze(self.y)
            self.y = self.sigmoid.forward(self.y)
            self.ys.append(self.y)
        return np.array(self.ys)

    def backward(self, error_tensor):
        # initialize the result error tensor
        result_e = np.empty((self.input_size, error_tensor.shape[0])).T
        # error tensor from the last hidden state
        error_h = np.zeros(self.hidden_size)  # hidden state from the last time step
        for i in range(error_tensor.shape[0] - 1, -1, -1):
            # -----> 1. compute the error vector from output y_t <-----
            error_vector = error_tensor[i]
            error_sigmoid = self.sigmoid.backward(error_vector)
            # fc layer backward
            the_h = self.hs[i + 1]
            self.y_fc.curr_input = np.expand_dims(the_h, axis=0)
            error_y_fc = self.y_fc.backward(np.expand_dims(error_sigmoid, axis=0))
            # reduce the first dimension
            error_y_fc = np.squeeze(error_y_fc)
            # -----> 2. compute the error vector from hidden state h_t+1 <-----
            error_tanh = self.tanh.backward(error_h)
            # fc layer backward
            the_x = self.xs[i]
            the_h = self.hs[i]
            the_x_n_h = np.concatenate((the_x, the_h))
            self.h_fc.curr_input = np.expand_dims(the_x_n_h, axis=0)
            error_h_fc = self.h_fc.backward(np.expand_dims(error_tanh, axis=0))
            # reduce the first dimension
            error_h_fc = np.squeeze(error_h_fc)
            # -----> 3. divide the concatenated error vector for x and h respectively <-----
            result_e[i] = error_h_fc[:self.input_size]  # x
            error_h = error_h_fc[self.input_size:]  # h
            # -----> sum up the copied error for h_t <-----
            error_h += error_y_fc
        return result_e

    def get_memorize(self):
        return self.memorize

    def set_memorize(self, m=True):
        self._memorize = m

    memorize = property(get_memorize, set_memorize)

    def initialize_components(self):
        self.h_fc = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.y_fc = FullyConnected(self.hidden_size, self.output_size)
        self.tanh = TanH()
        self.sigmoid = Sigmoid()

    def get_optimizer(self):
        return self.y_fc.optimizer, self.h_fc.optimizer

    def set_optimizer(self, optimizer):
        self.y_fc.optimizer = optimizer
        self.h_fc.optimizer = optimizer

    optimizer = property(get_optimizer, set_optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        self.y_fc.initialize(weights_initializer, bias_initializer)
        self.h_fc.initialize(weights_initializer, bias_initializer)

    def norm(self):
        return self.y_fc.norm() + self.h_fc.norm()

    @property
    def gradient_weights(self):
        return self.h_fc.gradient_weights

    def get_weights(self):
        return self.h_fc.weights

    def set_weights(self, w):
        self.h_fc.weights = w

    weights = property(get_weights, set_weights)
