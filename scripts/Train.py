from NeuralNetwork import NeuralNetwork
from DataUtils import DataUtils
class Train:
    def __init__(self):
        self.model = NeuralNetwork()
        self.dataUtils = DataUtils("./training_data/training_data.csv")
    def train(self):
        expectedOutput, inputData, numberOfSamples = self.dataUtils.prepare_data()
        lr = self.model.learning_rate
        for e in range(10000):
            y_hat, cache = self.model.feed_forward(inputData)
            if e % 20 == 0:
                print(f"Epoch: {e} Cost: {self.model.cost(y_hat, expectedOutput)}")

            dW3, db3, dA2 = self.model.backprop_layer_3(y_hat, expectedOutput, numberOfSamples, cache["activation2"])
            dW2, db2, dA1 = self.model.backprop_layer_2(dA2, cache["activation1"], cache["activation2"])
            dW1, db1 = self.model.backprop_layer_1(dA1, cache["activation0"], cache["activation1"])

            self.model.weight3 -= lr * dW3
            self.model.weight2 -= lr * dW2
            self.model.weight1 -= lr * dW1
            self.model.bias3 -= lr * db3
            self.model.bias2 -= lr * db2
            self.model.bias1 -= lr * db1
        self.model.save_model("./training_data/trained_model.npz")



