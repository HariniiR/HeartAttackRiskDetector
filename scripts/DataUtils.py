import pandas as pd
import numpy as np
class DataUtils:
    def __init__(self, filename):
        self.filename = filename
        self.layerSize = [3,4,4,1]
    def prepare_data(self):
        training_data = pd.read_csv(self.filename, skiprows=1)
        training_data = np.array(training_data)
        inputData = training_data[:, :-1]/300.0
        expectedOutput = training_data[:, -1]
        numberOfSamples = len(expectedOutput)
        return expectedOutput.reshape(self.layerSize[3], numberOfSamples), inputData.T, numberOfSamples

