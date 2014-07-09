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

def file2matrix(filename, extra):
    """
    ���ı���¼ת��ΪNumPy
    Args:
        filename: 
        extra:
    Returns:

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
    ��һ������ֵ
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals
