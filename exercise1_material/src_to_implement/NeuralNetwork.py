import copy


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.X = None
        self.y = None
        self.y_hat = None

    def forward(self):
        self.X, self.y = self.data_layer.next()
        # feed input through layers
        for layer in self.layers[:-1]:
            self.X = layer.forward(self.X)
        # get prediction by prediction layer
        self.y_hat = self.layers[-1].forward(self.X)
        # get loss value by loss layer
        loss = self.loss_layer.forward(self.y_hat, self.y)
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.y)
        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        # feed input through fully-connected layers
        for layer in self.layers[:-1]:
            input_tensor = layer.forward(input_tensor)
        # get prediction by prediction layer
        y_hat = self.layers[-1].forward(input_tensor)
        return y_hat
