import kNN
from numpy import *
import operator
from os import listdir

import trees

myDat, labels = trees.createDataSet()
myTree = trees.createTree(myDat, labels)
print myTree
