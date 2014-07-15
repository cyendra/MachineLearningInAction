import kNN
from numpy import *
import operator
from os import listdir
import trees
import treePlotter

import bayes

import logRegres

dataArr, labelMat = logRegres.loadDataSet("testSet.txt")
weights = logRegres.gradAscent(dataArr, labelMat)
logRegres.plotBestFit(dataArr, labelMat, weights.getA())

weights2 = logRegres.stocGradAscent0(dataArr, labelMat)
logRegres.plotBestFit(dataArr, labelMat, weights2)

weights3 = logRegres.stocGradAscent1(dataArr, labelMat)
logRegres.plotBestFit(dataArr, labelMat, weights3)