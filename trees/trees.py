from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 计算香农熵
def calcShannonEnt(dataSet):
    # 计算数据集中实例的总数
    numEntries = len(dataSet)
    # 创建一个数据字典，用它的键值记录最后一列的值
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in list(labelCounts):
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        # 计算类别出现的概率
        prob = float(labelCounts[key]) / numEntries
        # 计算香农熵
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# dataSet 待划分的数据集
# axis    划分数据集的特征
# value   特征的返回值
def splitDataSet(dataSet, axis, value):
    # 创建新的list对象
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    # 获取当前数据集特征属性个数
    numFeatures = len(dataSet[0]) - 1
    # 计算原始香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 原始信息增益值
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 创建唯一的分类标签列表(将数据集中所有第i个特征或者所有可能存在的值写入这个新list中)
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0

        # 遍历当前特征中的所有唯一属性值
        for value in uniqueVals:
            # 对每个特征划分一次数据集
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            # 计算新熵值
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 多数表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in list(classCount):
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# dataSet 数据集
# labels  标签列表:包含了数据集中所有特征的标签
def createTree(dataSet, labels):
    # 创建一个包含了数据集所有类标签的列表变量
    classList = [example[-1] for example in dataSet]

    # 第一个停止条件：所有的类标签完全相同
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 第二个停止条件：使用完所有特征，仍然不能将数据集划分成仅包含唯一类别的分组，挑选出现次数最多的类别作为返回指
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    # 创建树
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 复制类标签
        subLabels = labels[:]
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        myTree[bestFeatLabel][value] = createTree(subDataSet, subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree)[0]
    secondDict = inputTree[firstStr]

    # 将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    for key in list(inputTree):
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def test():
    myDat, labels = createDataSet()
    myTree = createTree(myDat, labels)
    storeTree(myTree, 'classifierStorage.txt')


def main(filename):
    fr = open(filename)
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    return lensesTree
