import operator
from os import listdir

from numpy import tile, array, zeros, shape


# inX: 用于分类的输入向量
# dataSet: 训练样本集
# labels: 标签向量
# k: 最邻近的数目
def classify(inX, dataSet, labels, k):
    # 获取矩阵的行数
    dataSetSize = dataSet.shape[0]
    # tile(inX, (dataSetSize, 1)) 将inX这个行向量复制为与dataSet同维度的矩阵
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 将矩阵里每一个元素乘以本身
    sqDiffMat = diffMat ** 2
    # 把矩阵的每一行相加，得到一个列向量
    sqDistances = sqDiffMat.sum(axis=1)
    # 将列向量的每一个元素开平方
    distances = sqDistances ** 0.5
    # 把列向量按照升序的形式进行排序，并以排序前各元素对应的索引的形式返回
    sortedDistIndicies = distances.argsort()
    return getBestLabel(k, labels, sortedDistIndicies)


def getBestLabel(k, labels, sortedDistIndicies):
    classCount = {}
    for i in range(k):
        # 获取对应元素中的标记
        voteIlabel = labels[sortedDistIndicies[i]]
        # 查询classCount中是否存在这个标记，如未存在，把该标记数的数目设为1，否则加1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 对classCount这个dict根据value( key=operator.itemgetter(1))降序（默认升序，因为reverse=True，所以降序）
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def fileToMatrix(fileName):
    # 打开文件
    fr = open(fileName)
    # 获取内容行数列表
    arrayOlines = fr.readlines()
    # 行数数目
    numberOfLines = len(arrayOlines)
    # 建立一个与内容同行3列的零矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # 遍历每行内容
    for line in arrayOlines:
        # 去除第i行内容的回车符
        line = line.strip()
        # 根据Tab符号划分成对应的数组
        listFromLine = line.split('\t')
        # 将划分后的数组的前三个元素赋值给零矩阵的第i行
        returnMat[index, :] = listFromLine[0:3]
        # 将划分后的数组的最后一个元素添加到Label数组中
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
        # 返回生成的矩阵和标签数组
    return returnMat, classLabelVector


# 归一化矩阵
def autoNorm(dataSet):
    # 返回矩阵中每列中最小元素组成的行向量
    minVals = dataSet.min(0)
    # 返回矩阵中没列中最大元素组成的行向量
    maxVals = dataSet.max(0)
    # 返回两个向量之差
    ranges = maxVals - minVals

    # 建立与dataSet同维度的零矩阵
    normDataSet = zeros(shape(dataSet))
    # 矩阵的行数
    m = dataSet.shape[0]

    # 原矩阵减去由最小值向量复制生成的同维度矩阵
    normDataSet = dataSet - tile(minVals, (m, 1))

    # 处理后的矩阵与最大最小差量复制而成的同维度矩阵的特征值相除
    normDataSet = normDataSet / tile(ranges, (m, 1))  # eigenvalue division
    return normDataSet, ranges, minVals


def datingClassTest(hoRatio=0.1, k=3):
    # 将文件转换为矩阵
    datingDataMat, datingLabels = fileToMatrix('datingTestSet.txt')
    # 归一化矩阵
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # normMat = datingDataMat
    # 获取矩阵的行数
    m = normMat.shape[0]
    # 测试算法的数据条数
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], k)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("the total error rate is %f" % (errorCount / float(numTestVecs)))
    return errorCount, numTestVecs


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    # 文件转换为矩阵
    datingDataMat, datingLabels = fileToMatrix('datingTestSet.txt')
    # 归一化矩阵
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 测试数据
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


def imgToVector(fileName):
    # 建立1024列行向量
    returnVect = zeros((1, 1024))
    # 打开32x32像素的文件
    fr = open(fileName)
    for i in range(32):
        # 读取第i行
        lineStr = fr.readline()
        for j in range(32):
            # 将第i行的第j个元素负责给行向量
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    trainingMat, hwLabels = getTrainingMatAndLabel()

    testFileNameList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileNameList)
    for i in range(mTest):
        fileNameStr = testFileNameList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = imgToVector('testDigits/%s' % fileNameStr)

        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


def getTrainingMatAndLabel():
    hwLabels = []
    # 获取trainingDigits文件夹下面的文件列表
    trainFileNameList = listdir('trainingDigits')
    m = len(trainFileNameList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 获取第i个文件
        fileNameStr = trainFileNameList[i]
        # 获取文件名
        fileStr = fileNameStr.split('.')[0]
        # 获取数字（Label）
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 将第i个文件的内容添加到矩阵中
        trainingMat[i, :] = imgToVector('trainingDigits/%s' % fileNameStr)
    return trainingMat, hwLabels


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def plotMat(datingDataMat, datingLabels):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()


def main():
    # group, labels = createDataSet()
    # return classify([0, 0], group, labels, 3)
    datingDataMat, datingLabels = fileToMatrix('datingTestSet.txt')
    plotMat(datingDataMat, datingLabels)

#
# # datingDataMat, datingLabels = fileToMatrix('datingTestSet2.txt')
# # normMat, ranges, minVals = autoNorm(datingDataMat)
# # return normMat, ranges, minVals
