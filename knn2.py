#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:42:48 2020

@author: ragezor
"""

import numpy as np
import matplotlib.pyplot as plt
import knn as K


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines, 4))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(',')
        returnMat[index, :] = listFromLine[0:4]
        if listFromLine[-1] == 'Iris-setosa':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'Iris-versicolor':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'Iris-virginica':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector
    return listFromLine


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    normDataSet = np.zeros(dataSet.shape)
    normDataSet = (dataSet - minVals) / (maxVals - minVals)
    return normDataSet


irisDataMat, irisLabels = file2matrix('iris.data.txt')

dataSet = autoNorm(irisDataMat)
print(dataSet)

m = 0.85
dataSize = dataSet.shape[0]
print(dataSize)
trainSize = int(m * dataSize)
testSize = int((1 - m) * dataSize)
print(trainSize, testSize)
k = 3
error = 0
for i in range(testSize):
    result = K.knn(dataSet[trainSize + i - 1, :], dataSet[0:trainSize, :], irisLabels[0:trainSize], k)
    if result != irisLabels[trainSize + i - 1]:
        error = error + 1

print("error:", error / testSize)