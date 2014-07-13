import kNN
from numpy import *
import operator
from os import listdir

import trees

import treePlotter
myDat, labels = trees.createDataSet()
print labels
myTree=treePlotter.retrieveTree(0)
print myTree
print trees.classify(myTree,labels,[1,0])
print trees.classify(myTree,labels,[1,1])