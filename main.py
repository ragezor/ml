
import numpy as np
from os import listdir
import knn as k


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# load the training set
hwLabels = []
trainingFileList = listdir('trainingDigits')

m = len(trainingFileList)
trainingMat = np.zeros((m, 1024))
for i in range(m):
    fileNameStr = trainingFileList[i]
    fileStr = fileNameStr.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    hwLabels.append(classNumStr)
    trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

# iterate through the test set
testFileList = listdir('testDigits')
errorCount = 0.0
mTest = len(testFileList)
for i in range(mTest):
    fileNameStr = testFileList[i]
    fileStr = fileNameStr.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
    classifierResult = k.knn(vectorUnderTest, trainingMat, hwLabels, 3)
    print("KNN得到的辨识结果是: %d, 实际值是: %d" % (classifierResult, classNumStr))
    if (classifierResult != classNumStr): errorCount += 1.0
print("\n辨识错误数量为: %d" % errorCount)
print("\n辨识率为: %f ％" % ((1 - errorCount / float(mTest)) * 100))