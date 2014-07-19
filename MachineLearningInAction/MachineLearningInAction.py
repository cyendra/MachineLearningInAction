import kNN
from numpy import *
import operator
from os import listdir
import trees
import treePlotter

import bayes

import logRegres

import svmMLiA
import boost
import adaboost

datMat, classLabels = adaboost.loadSimpData()

D = mat(ones((5,1))/5)
print boost.buildStump(datMat, classLabels, D)
