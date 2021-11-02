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

def kernFunc(x, y):
    mean = np.linalg.norm(x - y)**2
    variance = 1
    return np.exp(-mean/(2*variance))

def multiSVM(log, iters):
    weights = pegPercKernel(log.trainData, log.trainLabels, kernFunc, iters)
    errors = 0
    for i in range(len(log.testLabels[:500])):
        decision = 0
        for j in range(len(log.testLabels)):
            decision += weights[j] * log.trainLabels[j] * kernFunc(log.trainData[j], log.testData[i])
        if decision < 0:
            prediction = -1
        else:
            prediction = 1
        if prediction != log.testLabels[i]: errors += 1
    return 1 - errors/len(log.testLabels)

def pegSVM(x, y, iterations, lam=0.1):
    weights = np.zeros(x[0].shape)
    for i in range(iterations):
        iterCount = randint(0, len(y)-1)
        step = 1/(lam*(i+1))
        decision = y[iterCount] * weights @ x[iterCount].T
        if decision < 1:
            weights = (1 - step*lam) * weights + step*y[iterCount]*x[iterCount]
        else:
            weights = (1 - step*lam) * weights
    return weights

def pegPercKernel(x, y, kernel, iterations, lam=0.1):
    weights = np.zeros(len(y))
    for _ in range(iterations):
        print(f"Peg Iteration: {_}")
        it = randint(0, len(y)-1)
        decision = 0
        for j in range(len(y)):
            decision += weights[j] * y[it] * kernel(x[it], x[j])
        decision *= y[it]/lam
        if decision < 1:
            weights[it] += 1
    return weights

def testBase():
    log = DataSet()
    binaryTestTrim(log)
    log.splitData(8000)
    iters = 10
    kernel = False

    accuracy = multiSVM(log, iters)
    print('Accuracy Against Test:', accuracy)

def sepData(log, id):
    _log = {'data': [],'labels': []}

    for i in range(len(log.trainLabels)):
        _log['data'].append(log.trainData[i])
        if log.trainLabels[i] == id:
            _log['labels'].append(1)
        else:
            _log['labels'].append(-1)
    _log['data'] = np.array(_log['data'])
    _log['labels'] = np.array(_log['labels'])
    return _log

def testECOC():
    log = DataSet()
    log.splitData(7000)
    iters = 1000
    weightClasses = []
    for i in range(log.outputSize):
        print("using RBf")
        _log = sepData(log, i)
        weightClasses.append(pegPercKernel(_log['data'], _log['labels'], kernFunc, iters))
    errors = 0
    for i in range(len(log.testLabels)):
        predictions = []
        for k in range(10):
            weights = weightClasses[k]
            decision = 0
            for j in range(len(log.trainLabels)):
                decision += weights[j] * log.trainLabels[j] * kernFunc(log.trainData[j], log.testData[i])
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
    testBase()
    testECOC()

