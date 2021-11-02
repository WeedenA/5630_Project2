import numpy as np
import math
from random import randint
from manageDataset import DataSet
from pegasosPaper import *


def binaryTestTrim(log):
    xVals = []
    yVals = []
    for i in range(len(log.labels)):
        if log.labels[i] == 0:
            yVals.append(-1)
            xVals.append(log.data[i])
        elif log.labels[i] == 1:
            yVals.append(1)
            xVals.append(log.data[i])
        else:
            pass
    log.data = xVals
    log.labels = yVals


def linearSVM(log, iters):
    weights = pegSVM(log.trainData, log.trainLabels,iters)
    errors = 0
    for i in range(len(log.testLabels)):
        decision = weights @ log.testData[i].T
        if decision < 0:
            prediction = -1
        else:
            prediction = 1
        if prediction != log.testLabels[i]: errors += 1
    return 1 - errors/len(log.testLabels)

def kernel_function(x, y):
    mean = np.linalg.norm(x - y)**2
    variance = 1
    return np.exp(-mean/(2*variance))

def kernelPerceptron(log, iters):
    weights = pegPercKernel(log.trainData, log.trainLabels, kernel_function, iters)
    errors = 0
    for i in range(len(log.testLabels[:500])):
        decision = 0
        for j in range(len(log.testLabels)):
            decision += weights[j]*log.trainLabels[j]*kernel_function(log.trainData[j], log.testData[i])
        if decision < 0:
            prediction = -1
        else:
            prediction = 1
        if prediction != log.testLabels[i]: errors += 1
    return 1 - errors/len(log.testLabels)

def testBinary():
    log = DataSet()
    binaryTestTrim(log)
    log.splitData(8000)
    iters = 10
    kernel = False

    accuracy = kernelPerceptron(log, iters)
    print('Accuracy Against Test:', accuracy)

def send_data(log, id):
    dataset = {
        'data': [],
        'labels': []
    }

    for i in range(len(log.trainLabels)):
        dataset['data'].append(log.trainData[i])
        if log.trainLabels[i] == id:
            dataset['labels'].append(1)
        else:
            dataset['labels'].append(-1)
    dataset['data'] = np.array(dataset['data'])
    dataset['labels'] = np.array(dataset['labels'])
    return dataset

def testMultiClass():
    log = DataSet()
    log.splitData(3500)
    iters = 100
    weightClasses = []

    for i in range(log.outputSize):
        print("using RBf")
        _log = send_data(log, i)
        weightClasses.append(pegPercKernel(_log['data'], _log['labels'], kernel_function, iters))
    errors = 0
    for i in range(len(log.testLabels)):
        predictions = []
        for k in range(10):
            weights = weightClasses[k]
            decision = 0
            for j in range(len(log.trainLabels)):
                decision += weights[j] * log.trainLabels[j] * kernel_function(log.trainData[j], log.testData[i])
            predictions.append(decision)
        predictions = np.array(predictions)
        classLabels = predictions.argmax()
        if classLabels != log.testLabels[i]:
            errors += 1
            print(f"Error: {classLabels} found, {log.testLabels[i]} expected, index {i}")
    accuracy = 1 - errors / len(log.testLabels)
    print(f"Error: {errors / len(log.testLabels)}")
    print(f"Accuracy: {accuracy}")




if __name__ == "__main__":

    testMultiClass()

