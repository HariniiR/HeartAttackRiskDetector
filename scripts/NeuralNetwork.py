import numpy as np

np.random.seed(0)

def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))

class NeuralNetwork:
    def __init__(self):
        self.layerSize = [3, 4, 4, 1]

        self.weight1 = np.random.randn(self.layerSize[1], self.layerSize[0])
        self.weight2 = np.random.randn(self.layerSize[2], self.layerSize[1])
        self.weight3 = np.random.randn(self.layerSize[3], self.layerSize[2])

        self.bias1 = np.random.randn(self.layerSize[1], 1)
        self.bias2 = np.random.randn(self.layerSize[2], 1)
        self.bias3 = np.random.randn(self.layerSize[3], 1)

        self.learning_rate = 0.01

    def feed_forward(self, inputDataTransposed):
        z1 = self.weight1 @ inputDataTransposed + self.bias1
        activation1 = sigmoid(z1)
        z2 = self.weight2 @ activation1 + self.bias2
        activation2 = sigmoid(z2)
        z3 = self.weight3 @ activation2 + self.bias3
        activation3 = sigmoid(z3)

        cache = {"activation0": inputDataTransposed, "activation1": activation1, "activation2": activation2, "activation3": activation3}
        return activation3, cache

    def cost(self, y_hat, y):
        numberOfSamples = y_hat.shape[1]
        losses = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return (1 / numberOfSamples) * np.sum(losses)

    def backprop_layer_3(self, y_hat, y, numberOfSamples, activation2):
        err = (1 / numberOfSamples) * (y_hat - y)
        return err @ activation2.T, np.sum(err, axis=1, keepdims=True), self.weight3.T @ err

    def backprop_layer_2(self, error, a1, activation2):
        delta = error * activation2 * (1 - activation2)
        return delta @ a1.T, np.sum(delta, axis=1, keepdims=True), self.weight2.T @ delta

    def backprop_layer_1(self, error, activation0, activation1):
        delta = error * activation1 * (1 - activation1)
        return delta @ activation0.T, np.sum(delta, axis=1, keepdims=True)
    def save_model(self, filename):
        np.savez(filename,
                 weight1=self.weight1,
                 bias1=self.bias1,
                 weight2=self.weight2,
                 bias2=self.bias2,
                 weight3=self.weight3,
                 bias3=self.bias3)


    @staticmethod
    def load_model(filename):
        data = np.load(filename)

        model = NeuralNetwork()  # using the default constructor

        model.weight1 = data['weight1']
        model.bias1 = data['bias1']
        model.weight2 = data['weight2']
        model.bias2 = data['bias2']
        model.weight3 = data['weight3']
        model.bias3 = data['bias3']

        return model




