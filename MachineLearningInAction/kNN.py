# coding=gbk
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    """
    k-�ڽ��㷨
    Args:
        inX: ���ڷ������������
        dataSet: �����ѵ��������
        labels: ��ǩ����
        k: ����ѡ�����ڽ�����Ŀ
    Returns:
        ����Ƶ����ߵ�Ԫ�ر�ǩ
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
