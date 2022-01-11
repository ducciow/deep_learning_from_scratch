class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.testing_phase = False

    def forward(self, input_tensor):
        pass

    def backward(self, error_tensor):
        pass

    def norm(self):
        return 0
