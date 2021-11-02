import numpy as np
import operator
import time
from manageDataset import DataSet
from sklearn.metrics import confusion_matrix
import pickle


def sig(x, deriv=False):
    if (deriv==True):
        return sig(x) * (1 - sig(x))
    return 1/(1+np.exp(-x))

def train():
    startTime = time.time()
    X = log.trainData
    y = log.trainLabels

    np.random.seed(1)

    web1 = 2 * np.random.random((780,100)) - 1
    web2 = 2 * np.random.random((100,10)) - 1
    bias = 2 * np.random.random((2,1)) - 1
    np.seterr(all='ignore')
    for i in range(1000):
        inputChain = X
        hiddenChain = sig(np.dot(inputChain, web1) + bias[0][0])
        outputChain = sig(np.dot(hiddenChain, web2) + bias[1][0])

        outputError = y - outputChain
        if (i%100) == 0:
            print(f"Iteration {i} Error: {float(np.mean(np.abs(outputError)))}")

        outputChange = outputError * sig(outputChain, deriv=True)
        hiddenError = outputChange.dot(web2.T)
        hiddenChange = hiddenError * sig(hiddenChain, deriv=True)
        biasChange1 = np.mean(hiddenError) * sig(bias[0][0], deriv=True)
        biasChange2 = np.mean(outputError) * sig(bias[1][0], deriv=True)

        web2 += hiddenChain.T.dot(outputChange)
        web1 += inputChain.T.dot(hiddenChange)
        bias[0][0] *= biasChange1
        bias[1][0] *= biasChange2
    log.inputWeights = web1
    log.outputWeights = web2
    print(f"Finished Training, took {time.time() - startTime} seconds")

def predict(predData, trueLabels):
    i = 0
    correct = 0
    realLabels = []
    predLabels = []
    for item in predData:
        log.hiddenValues = sig(np.dot(item, log.inputWeights))
        log.outputValues = sig(np.dot(log.hiddenValues, log.outputWeights))
        index, value = max(enumerate(log.outputValues), key=operator.itemgetter(1))
        indexTrue, valueTrue = max(enumerate(trueLabels[i]), key=operator.itemgetter(1))
        predLabels.append(index)
        realLabels.append(indexTrue)
        #print(f"Prediction: {index} vs Actual: {indexTrue}")
        if index == indexTrue:
            correct += 1
        i += 1
    print(f"Accuracy: {correct/len(predData)}")
    confMatrix = confusion_matrix(realLabels, predLabels)
    print(confMatrix)
    with open('conf.txt', 'wb') as f:
        pickle.dump(confMatrix, f)
    f.close()

if __name__ == "__main__":
    log = DataSet()
    log.setLabels()
    log.splitData(42000)
    print("Training...")
    train()
    print("Accuracy against Training Data")
    predict(log.trainData, log.trainLabels)
    print("Accuracy against Test Data")
    predict(log.testData, log.testLabels)


