import numpy as np
from sklearn.datasets import load_svmlight_file
import pickle

class DataSet(object):
    def __init__(self):

        self.data, self.labels = self.readPickles()
        self.testData, self.testLabels = [], []
        self.trainData, self.trainLabels = [], []
        self.inputSize = len(self.data[0])
        self.hiddenSize = 100
        self.outputSize = 10

        self.hiddenValues = []
        self.outputValues = []
        self.inputWeights = []
        self.outputWeights = []
        self.biases = np.random.rand(2,1)

    # must have initialized dataset with openData()
    def readPickles(self):
        with open('data.scale', 'rb') as f:
            data = pickle.load(f)
        f.close()
        with open('label.scale', 'rb') as f:
            label = pickle.load(f)
        f.close()
        return data, label

    def openData(self):
        x, label = load_svmlight_file('MnistORIG.scale')
        x= x.toarray()
        label = label.astype(int)
        return x, label

    def setLabels(self):
        oldLabels = self.labels
        self.labels = np.zeros((len(oldLabels), 10))
        for i in range(len(oldLabels)):
            index = oldLabels[i]
            self.labels[i][index] = 1

    def splitData(self, splitAt):
        self.testData, self.testLabels = self.data[splitAt:], self.labels[splitAt:]
        self.trainData, self.trainLabels = self.data[:splitAt], self.labels[:splitAt]

