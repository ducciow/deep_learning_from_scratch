import copy
import numpy as np
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
        self.xs = None  # length: time steps
        self.hs = []  # length: time steps + 1
        # composed layers
        self.h_fc = None
        self.y_fc = None
        self.tanh = None
        self.sigmoid = None
        # optimizers
        self.has_optimizer = False
        self._optimizer_weights_h = None
        self._optimizer_bias_h = None
        self._optimizer_weights_y = None
        self._optimizer_bias_y = None
        # weight gradients
        self._gradient_weights_h = None
        self._gradient_bias_h = None
        self._gradient_weights_y = None
        self._gradient_bias_y = None
        # initialize composed layers
        self.initialize_components()
        # initialize gradients
        self.initialize_weight_gradients()

    def forward(self, input_tensor):
        if not self._memorize:
            self.h = np.zeros(self.hidden_size)
        self.xs = input_tensor.copy()
        self.hs = [self.h]
        self.initialize_weight_gradients()
        ys = []
        # iterate through time
        for x in self.xs:
            # -----> 1. compute h_t <-----
            # concatenate h_t-1 and x_t
            h_n_x = np.concatenate((self.h, x))
            # fc layer
            self.h = np.squeeze(self.h_fc.forward(np.expand_dims(h_n_x, axis=0)))
            # activation layer
            self.h = self.tanh.forward(self.h)
            # cache
            self.hs.append(self.h)
            # -----> 2. compute y_t <-----
            # fc layer
            y = np.squeeze(self.y_fc.forward(np.expand_dims(self.h, axis=0)))
            # activation layer
            y = self.sigmoid.forward(y)
            # cache
            ys.append(y)
        return np.array(ys)

    def backward(self, error_tensor):
        # initialize the result error tensor
        result_e = np.empty(self.xs.shape)
        # error tensor from the last hidden state
        error_h = np.zeros(self.hidden_size)
        # iterate reversely through time
        for i in range(error_tensor.shape[0] - 1, -1, -1):

            # -----> 1. compute the error vector from output y_t <-----
            error_vector = error_tensor[i].copy()
            # activation layer backward
            error_sigmoid = self.sigmoid.backward(error_vector)
            # fc layer backward
            the_h = self.hs[i + 1]
            self.y_fc.curr_input = np.expand_dims(the_h, axis=0)
            error_y_fc = np.squeeze(self.y_fc.backward(np.expand_dims(error_sigmoid, axis=0)))

            # -----> 2. sum up the copied error for h_t <-----
            error_h += error_y_fc

            # -----> 3. compute the error vector from hidden state h_t+1 <-----
            # activation layer backward
            error_tanh = self.tanh.backward(error_h)
            # fc layer backward
            the_h = self.hs[i]
            the_x = self.xs[i]
            the_h_n_x = np.concatenate((the_h, the_x))
            self.h_fc.curr_input = np.expand_dims(the_h_n_x, axis=0)
            error_h_fc = np.squeeze(self.h_fc.backward(np.expand_dims(error_tanh, axis=0)))

            # -----> 4. divide the concatenated error vector for x and h respectively <-----
            error_h = error_h_fc[:self.hidden_size]  # h
            result_e[i] = error_h_fc[self.hidden_size:]  # x

            # -----> 5. accumulate gradients
            self._gradient_weights_h += self.h_fc.gradient_weights
            self._gradient_bias_h += self.h_fc.gradient_bias
            self._gradient_weights_y += self.y_fc.gradient_weights
            self._gradient_bias_y += self.y_fc.gradient_bias

        if self.has_optimizer:
            self.update_weights()

        return result_e

    def initialize_components(self):
        self.h_fc = FullyConnected(self.hidden_size + self.input_size, self.hidden_size)
        self.y_fc = FullyConnected(self.hidden_size, self.output_size)
        self.tanh = TanH()
        self.sigmoid = Sigmoid()

    def initialize_weight_gradients(self):
        self._gradient_weights_h = np.zeros((self.hidden_size + self.input_size, self.hidden_size))
        self._gradient_bias_h = np.zeros(self.hidden_size)
        self._gradient_weights_y = np.zeros((self.hidden_size, self.output_size))
        self._gradient_bias_y = np.zeros(self.output_size)

    def update_weights(self):
        self.h_fc.weights = self._optimizer_weights_h.calculate_update(self.h_fc.weights, self._gradient_weights_h)
        self.h_fc.bias = self._optimizer_bias_h.calculate_update(self.h_fc.bias, self._gradient_bias_h)
        self.y_fc.weights = self._optimizer_weights_y.calculate_update(self.y_fc.weights, self._gradient_weights_y)
        self.y_fc.bias = self._optimizer_bias_y.calculate_update(self.y_fc.bias, self._gradient_bias_y)

    def initialize(self, weights_initializer, bias_initializer):
        self.y_fc.initialize(weights_initializer, bias_initializer)
        self.h_fc.initialize(weights_initializer, bias_initializer)

    def norm(self):
        norm = 0
        if self._optimizer_weights_h.regularizer:
            norm += self._optimizer_weights_h.regularizer.norm(self.h_fc.weights)
        if self._optimizer_bias_h.regularizer:
            norm += self._optimizer_bias_h.regularizer.norm(self.h_fc.bias)
        if self._optimizer_weights_y.regularizer:
            norm += self._optimizer_weights_y.regularizer.norm(self.y_fc.weights)
        if self._optimizer_bias_y.regularizer:
            norm += self._optimizer_bias_y.regularizer.norm(self.y_fc.bias)
        return norm

    def get_memorize(self):
        return self.memorize

    def set_memorize(self, m=True):
        self._memorize = m

    memorize = property(get_memorize, set_memorize)

    def get_optimizer(self):
        return self._optimizer_weights_h, self._optimizer_bias_h, self._optimizer_weights_y, self._optimizer_bias_y

    def set_optimizer(self, optimizer):
        self.has_optimizer = True
        self._optimizer_weights_h = copy.deepcopy(optimizer)
        self._optimizer_bias_h = copy.deepcopy(optimizer)
        self._optimizer_weights_y = copy.deepcopy(optimizer)
        self._optimizer_bias_y = copy.deepcopy(optimizer)

    optimizer = property(get_optimizer, set_optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_weights_h

    @property
    def weights(self):
        return self.h_fc.weights

    @weights.setter
    def weights(self, w):
        self.h_fc.weights = w









