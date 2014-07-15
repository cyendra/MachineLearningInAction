
def loadDataSet(filename):
    """
    ���ļ��ж�ȡ����
    Returns:
        dataMat: ���ݼ�
        labelMat: ��ǩ��
    """
    dataMat = [] # ���ݼ�
    labelMat = [] # ��ǩ��
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split() # �ָ�����
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # ����ǰ����ֵ�������ݼ���Ϊ��������һ����Ϊ 1
        labelMat.append(int(lineArr[2])) # ������ֵ���ڱ�ǩ��
    return dataMat, labelMat # �������ݼ����ǩ��

def sigmoid(inX):
    """
    Sigmoid ����
    """
    return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    """
    �ݶ�����
    Args:
        dataMatIn: ���ݼ�����άNumPy���飬ÿ�д���ͬ��������ÿ�д���ѵ������
        classLabels: ��ǩ��������ǩ����һ��һά����
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
