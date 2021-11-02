import numpy as np
from manageDataset import DataSet

class PolynomialPerceptron():
    def __init__(self, inputs, targets, n, p, max_iter):
        self.inputs = inputs
        self.targets = targets
        self.p = p
        self.n = n
        self.max_iter = max_iter
        self.alpha = np.zeros(n)

    def polyFunc(self, X, Y, p):
        Y = Y.T
        res = np.dot(X,Y)
        res = np.add(res, Y)
        res = np.exp(res, p)
        return res

    def train(self):
        K = self.polyFunc(self.inputs, self.inputs, self.p)
        for i in range(self.max_iter):
            print(f"Iteration {i}")
            for j in range(self.n):
                alph = self.alpha * self.targets
                k = alph * K[j]
                if self.targets[j] * k <= 0:
                    self.alpha[j] += 1

    def predict(self, inputData):
        tmp = (self.alpha * self.targets) * inputData
        testPred = 1 if tmp > 0 else 0
        return testPred

    def acc(self, inputs, targets):
        correct = 0
        kVal = self.polyFunc(inputs, self.inputs, self.p)
        for idx, each in enumerate(inputs):
            correct += self.predict(kVal[idx]) == targets[idx]
        return correct / len(inputs)

    def ECOC(self):
        pass

log = DataSet()
log.splitData(42000)
iters = 1000
n = 42000
c = 0.01

model = PolynomialPerceptron(log.trainData, log.trainLabels,n,c,iters)
model.train()
acc = model.acc(log.testData, log.testLabels)
print(f"accuracy: {acc}")
confMat = confusion_matrix(log.realLabels, log.testLabels)
with open('percConf.txt', 'wb') as f:
    pickle.dump(confMat, f)
f.close()