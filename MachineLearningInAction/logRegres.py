# coding=gbk

from numpy import *

def loadDataSet(filename):
    """
    从文件中读取数据
    Returns:
        dataMat: 数据集
        labelMat: 标签集
    """
    dataMat = [] # 数据集
    labelMat = [] # 标签集
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split() # 分割数据
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # 输入前两个值属于数据集，为方便计算第一列置为 1
        labelMat.append(int(lineArr[2])) # 第三个值属于标签集
    return dataMat, labelMat # 返回数据集与标签集

def sigmoid(inX):
    """
    Sigmoid 函数
    """
    return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    """
    梯度上升算法
    Args:
        dataMatIn: 数据集，二维矩阵，每列代表不同的特征，每行代表训练样本
        classLabels: 标签集，类别标签，是一个一维向量
    Returns:
        weights: 回归系数
    """
    dataMatrix = mat(dataMatIn) # 数据样本矩阵
    labelMat = mat(classLabels).transpose() # 标签矩阵的转置
    m, n = shape(dataMatrix) # 读取矩阵的大小 m * n，m 是行数，n是列数
    alpha = 0.001 # 向目标移动的步长
    maxCycles = 500 # 迭代次数
    weights = ones((n, 1)) # 创建 n*1 的数组
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights) # 矩阵相乘，得到一个 m * 1 的矩阵
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error # 梯度上升算法的迭代公式
    return weights

def plotBestFit(dataMatIn, classLabels, wei):
    """
    画出数据集和Logistic回归最佳拟合直线的函数
    """
    import matplotlib.pyplot as plt
    weights = wei.getA()
    dataMat, labelMat = dataMatIn, classLabels
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()






