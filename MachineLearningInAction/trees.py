# coding=gbk

from math import log
import operator

def majorityCnt(classList):
    """
    返回出现频率最高的标签名
    Args:
        classList: 标签集
    Returns:
        出现频率最高的标签名
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    """
    计算给定数据集的香农熵
    Args:
        dataSet: 数据集最后一列为标签
    Returns:
        shannonEnt: 香农熵
    """
    numEntries = len(dataSet) # 数据总数
    labelCounts = {} # 标签集
    for featVec in dataSet: # 遍历所有数据
        currentLabel = featVec[-1] # 获取数据的标签
        if currentLabel not in labelCounts.keys(): # 不存在该标签
            labelCounts[currentLabel] = 0 # 初始化标签数为0
        labelCounts[currentLabel] += 1 # 当前数据所从属的标签数量+1
    shannonEnt = 0.0 # 香农熵初值
    for key in labelCounts: # 遍历所有的标签
        prob = float(labelCounts[key])/numEntries # 选到该标签的概率
        shannonEnt -= prob * log(prob,2) # 香农熵公式
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    Args:
        dataSet: 带划分的数据集
        axis: 划分数据集的特征
        value: 需要返回的特征的值
    Returns:
        retDataSet: 抽取出的符合要求的元素集
    """
    retDataSet = []
    for featVec in dataSet: # 遍历每个数据
        if featVec[axis] == value: # 符合要求的数据
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:]) # 抽取出其他特征
            retDataSet.append(reducedFeatVec) # 加入结果集
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    Args:
        dataSet: 数据集
    Returns:
        bestFeature: 最好划分方式的特征编号
    """
    numFeatures = len(dataSet[0]) - 1 # 特征数
    baseEntropy = calcShannonEnt(dataSet) # 香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures): # 遍历所有特征
        featList = [example[i] for example in dataSet] # 所有数据特征i的集合
        uniqueVals = set(featList) # 将特征的不同取值放入集合
        newEntropy = 0.0
        for value in uniqueVals: # 对于特征的所有取值
            subDataSet = splitDataSet(dataSet, i, value) # 划分数据集
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet) # 香农熵
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain): # 更优的划分方式
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def createTree(dataSet, labels):
    """
    创建树
    Args:
        dataSet: 数据集
        labels: 特征标签集
    Returns:
        决策树
    """
    classList = [example[-1] for example in dataSet] # 提取出所有的类标签
    if classList.count(classList[0]) == len(classList): # 所有元素属于同一类
        return classList[0]
    if len(dataSet[0]) == 1: # 数据只有一个特征
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) # 最好的划分方式
    bestFeatLabel = labels[bestFeat] # 最好的划分特征所属特征标签
    myTree = {bestFeatLabel:{}} # 树
    del(labels[bestFeat]) # 删除最佳特征的名字
    featValues = [example[bestFeat] for example in dataSet] # 数据集中的该特征的值
    uniqueVals = set(featValues) # 值域
    for value in uniqueVals: # 对所有可能的取值
        subLabels = labels[:] # 子标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels) #递归求子树
    return myTree

#-----------------------------------------------

def classify(inputTree, featLabels, testVec):
    """
    使用决策树的分类函数
    Args:
        inputTree: 决策树
        featLabels: 标签向量
        testVec: 待分类的数据
    Returns:
        classLabel: 分类结果
    """
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    """
    储存决策树
    Args:
        inputTree: 待储存的决策树
        filename: 储存文件名
    """
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    """
    读取决策树
    Args:
        filename: 待读取的文件名
    """
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def getNumLeafs(myTree):
    """
        获取树的叶结点总数
    """
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else: numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    """
    获取树的深度
    """
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else: thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth
