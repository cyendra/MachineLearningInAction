import kNN
from numpy import *
import operator
from os import listdir
"""
group, labels = kNN.createDataSet()
print group
print labels
print kNN.classify0([0,0], group, labels, 3)
"""
"""
datingDataMat, datingLabels = kNN.file2matrix("datingTestSet2.txt", 3)
#print datingDataMat
#print datingLabels
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print normMat
print ranges
print minVals
"""

"""
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()

"""

def make_watcher():
    have_seen=[]
    def have_seen_watcher(x):
        if x in have_seen:
            return True
        else:
            have_seen.append(x)
            return False
    return have_seen_watcher

watcher=make_watcher()
print watcher(10)
print watcher(10)

def fun():
    n = []
    def res(x):
        n.append(x)
        return n
    return res

f = fun()
print f(2)
print f(3)









