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