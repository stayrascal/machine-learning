import matplotlib.pyplot as plt

# 定义文本框和箭头符号(定义树节点格式的常量)
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 准备数据
def retrieveTree(i):
    listOfTrees = listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                                 {'no surfacing': {0: 'no',
                                                   1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]


# 绘制带箭头的注解
def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction', xytext=centerPt,
                            textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot1():
    fig = plt.figure(1, facecolor='white')  # 创建一个新图形
    fig.clf()  # 清空绘图区
    createPlot.ax1 = plt.subplot(111, frameon=False)  # 全局变量
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


# 获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree)[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxPath = 0
    firstStr = list(myTree)[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxPath:
            maxPath = thisDepth
    return maxPath


# 在父子节点间填充文本信息
def plotMidText(centerPt, parentPt, txtString):
    xMid = (parentPt[0] - centerPt[0]) / 2.0 + centerPt[0]
    yMid = (parentPt[1] - centerPt[1]) / 2.0 + centerPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeText):
    # 计算宽和高
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)

    firstStr = list(myTree)[0]

    # 通过计算树所有叶子节点数，划分图形的高度，从而计算当前节点的中心位置
    centerPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)

    # 计算父节点和子节点的中间位置，并在此处添加文本标签信息
    plotMidText(centerPt, parentPt, nodeText)
    plotNode(firstStr, centerPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]

    # 按比例减少全局变量
    plotTree.yOff -= 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], centerPt, str(key))
        else:
            plotTree.xOff += 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), centerPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), centerPt, str(key))
    plotTree.yOff += 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axpros = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axpros)

    # 使用下列两个变量计算树节点的摆放位置，这样可以将树绘制在水平方向和垂直方向的中心位置
    plotTree.totalW = float(getNumLeafs(inTree))  # 存储树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))  # 存储树的高度

    # 追踪已经绘制的节点位置，以及放置下一个节点的恰当位置
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0

    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
