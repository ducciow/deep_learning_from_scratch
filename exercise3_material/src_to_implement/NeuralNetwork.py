import copy


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.X = None
        self.y = None
        self.y_hat = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        self.X, self.y = self.data_layer.next()
        # feed input through layers  # and sum up constraints
        norms = 0
        for layer in self.layers[:-1]:
            self.X = layer.forward(self.X)
            norms += layer.norm()
        # get prediction by prediction layer
        self.y_hat = self.layers[-1].forward(self.X)
        # get loss value by loss layer
        loss = self.loss_layer.forward(self.y_hat, self.y) + norms
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.y)
        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = self.optimizer
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = False
        for _ in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        # set testing_phase to be true
        self.phase = True
        # feed input through fully-connected layers
        for layer in self.layers[:-1]:
            input_tensor = layer.forward(input_tensor)
        # get prediction by prediction layer
        y_hat = self.layers[-1].forward(input_tensor)
        return y_hat

    def get_phase(self):
        return [(layer, layer.testing_phase) for layer in self.layers]

    def set_phase(self, is_test):
        for layer in self.layers:
            layer.testing_phase = is_test

    phase = property(get_phase, set_phase)