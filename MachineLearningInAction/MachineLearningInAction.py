import kNN
from numpy import *
import operator
from os import listdir
import trees
import treePlotter

import bayes

listOPosts, listClasses = bayes.loadDataSet()

myVocabList = bayes.createVocabList(listOPosts)
print myVocabList

print bayes.setOfWords2Vec(myVocabList, listOPosts[0])

print bayes.setOfWords2Vec(myVocabList, listOPosts[3])

trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))

p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
print pAb
print p0V
print p1V

print "--------------------"

bayes.testingNB()