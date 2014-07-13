# coding=gbk

from math import log
import operator

def majorityCnt(classList):
    """
    ���س���Ƶ����ߵı�ǩ��
    Args:
        classList: ��ǩ��
    Returns:
        ����Ƶ����ߵı�ǩ��
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
    ����������ݼ�����ũ��
    Args:
        dataSet: ���ݼ����һ��Ϊ��ǩ
    Returns:
        shannonEnt: ��ũ��
    """
    numEntries = len(dataSet) # ��������
    labelCounts = {} # ��ǩ��
    for featVec in dataSet: # ������������
        currentLabel = featVec[-1] # ��ȡ���ݵı�ǩ
        if currentLabel not in labelCounts.keys(): # �����ڸñ�ǩ
            labelCounts[currentLabel] = 0 # ��ʼ����ǩ��Ϊ0
        labelCounts[currentLabel] += 1 # ��ǰ�����������ı�ǩ����+1
    shannonEnt = 0.0 # ��ũ�س�ֵ
    for key in labelCounts: # �������еı�ǩ
        prob = float(labelCounts[key])/numEntries # ѡ���ñ�ǩ�ĸ���
        shannonEnt -= prob * log(prob,2) # ��ũ�ع�ʽ
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """
    ���ո��������������ݼ�
    Args:
        dataSet: �����ֵ����ݼ�
        axis: �������ݼ�������
        value: ��Ҫ���ص�������ֵ
    Returns:
        retDataSet: ��ȡ���ķ���Ҫ���Ԫ�ؼ�
    """
    retDataSet = []
    for featVec in dataSet: # ����ÿ������
        if featVec[axis] == value: # ����Ҫ�������
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:]) # ��ȡ����������
            retDataSet.append(reducedFeatVec) # ��������
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    ѡ����õ����ݼ����ַ�ʽ
    Args:
        dataSet: ���ݼ�
    Returns:
        bestFeature: ��û��ַ�ʽ���������
    """
    numFeatures = len(dataSet[0]) - 1 # ������
    baseEntropy = calcShannonEnt(dataSet) # ��ũ��
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures): # ������������
        featList = [example[i] for example in dataSet] # ������������i�ļ���
        uniqueVals = set(featList) # �������Ĳ�ͬȡֵ���뼯��
        newEntropy = 0.0
        for value in uniqueVals: # ��������������ȡֵ
            subDataSet = splitDataSet(dataSet, i, value) # �������ݼ�
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet) # ��ũ��
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain): # ���ŵĻ��ַ�ʽ
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def createTree(dataSet, labels):
    """
    ������
    Args:
        dataSet: ���ݼ�
        labels: ������ǩ��
    Returns:
        ������
    """
    classList = [example[-1] for example in dataSet] # ��ȡ�����е����ǩ
    if classList.count(classList[0]) == len(classList): # ����Ԫ������ͬһ��
        return classList[0]
    if len(dataSet[0]) == 1: # ����ֻ��һ������
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) # ��õĻ��ַ�ʽ
    bestFeatLabel = labels[bestFeat] # ��õĻ�����������������ǩ
    myTree = {bestFeatLabel:{}} # ��
    del(labels[bestFeat]) # ɾ���������������
    featValues = [example[bestFeat] for example in dataSet] # ���ݼ��еĸ�������ֵ
    uniqueVals = set(featValues) # ֵ��
    for value in uniqueVals: # �����п��ܵ�ȡֵ
        subLabels = labels[:] # �ӱ�ǩ
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels) #�ݹ�������
    return myTree

#-----------------------------------------------

def classify(inputTree, featLabels, testVec):
    """
    ʹ�þ������ķ��ຯ��
    Args:
        inputTree: ������
        featLabels: ��ǩ����
        testVec: �����������
    Returns:
        classLabel: ������
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
    ���������
    Args:
        inputTree: ������ľ�����
        filename: �����ļ���
    """
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    """
    ��ȡ������
    Args:
        filename: ����ȡ���ļ���
    """
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def getNumLeafs(myTree):
    """
        ��ȡ����Ҷ�������
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
    ��ȡ�������
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
