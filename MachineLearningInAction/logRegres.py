
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
    梯度上升
    Args:
        dataMatIn: 数据集，二维NumPy数组，每列代表不同的特征，每行代表训练样本
        classLabels: 标签集，类别标签，是一个一维向量
    """
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights - alpha * dataMatrix.transpose() * error
    return weights
