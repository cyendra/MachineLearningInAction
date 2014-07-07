# coding=gbk
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

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
    #print "dataSetSize=", dataSetSize
    #print "tile=", tile(inX, (dataSetSize, 1))
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    #print "diffMat=", diffMat
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #print "distances=", distances
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0) + 1
    #print "classCount", classCount
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
