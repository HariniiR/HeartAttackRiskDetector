from NeuralNetwork import NeuralNetwork
from Train import Train
import numpy as np
class Predict:
    def __init__(self):
        self.model = NeuralNetwork.load_model("trained_model.npz")
    def predict(self, inputs):
        inputs = np.array(inputs) / 300.0
        inputs_T = inputs.T
        output, _ = self.model.feed_forward(inputs_T)
        return output, np.round(output)
