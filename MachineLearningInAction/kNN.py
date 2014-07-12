# coding=gbk
from numpy import *
import operator

def classify0(inX, dataSet, labels, k):
    """
    k-邻近算法
    Args:
        inX: 用于分类的输入向量
        dataSet: 输入的训练样本集
        labels: 标签向量
        k: 用于选择最邻近的数目
    Returns:
        发生频率最高的元素标签
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename, extra):
    """
    将文本记录转换为NumPy
    Args:
        filename: 文件名
        extra: 特征数
    Returns:
        样本集, 标签集
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, extra))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0: extra]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
    归一化特征值
    Args:
        dataSet: 样本集
    Returns:
        新的样本矩阵
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals



def classTest(filename, extra, K):
    """
    分类器测试代码
    Args:
        filename: 文本记录的文件名
        extra: 特征数
        K: 用于选择最邻近的数目
    """
    hoRatio = 0.10
    dataMat, labels = file2matrix(filename, extra)
    normMat, ranges, minVals = autoNorm(dataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:], labels[numTestVecs:m], K)
        #print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, labels[i])
        if (classifierResult != labels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))


def img2vector(filename, N, M):
    """
    将N*M的二维矩阵转换为一维向量
    Args:
        filename: 由01串组成N行M列的文本文件
        N: 行数
        M: 列数
    """
    returnVect = zeros((1,N*M))
    fr = open(filename)
    for i in range(N):
        lineStr = fr.readlin()
        for j in range(M):
            returnVect[0,M*i+j] = int(lineStr[j])
    return returnVect


