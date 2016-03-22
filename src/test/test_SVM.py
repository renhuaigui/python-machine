# -*- coding:utf-8 -*-
'''
@Created on 2015年99月9日
@function: 
@author: Ren Huaigui
'''
from numpy import *  

import SVM  
from readData import readData
################## test svm #####################  
## step 1: load data  
print "step 1: load data..."  

# fileIn = open('D:/Desktop/python study/Image/src/testSet.txt')  
# for line in fileIn.readlines():  
#     lineArr = line.strip().split() 
#     print(lineArr) 
#     dataSet.append([float(tk) for tk in lineArr[:-1]])  
#     labels.append(float(lineArr[-1]))  

dataSet,labels = readData("E:/Desktop/Image/SVMData/gender_wechat.txt")
t = int(len(labels)/5)
dataSet = mat(dataSet)  
labels = mat(labels).T  
train_x = dataSet[0:t, :]  
train_y = labels[0:t, :]  
test_x = dataSet[t:len(labels), :]  
test_y = labels[t:len(labels), :]  
 
## step 2: training...  
print "step 2: training..."  
C = 0.6  
toler = 0.001  
maxIter = 50  
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('rbf', 0))  
  
## step 3: testing  
print "step 3: testing..."  
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)  
  
## step 4: show the result  
print "step 4: show the result..."    
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)  
#SVM.showSVM(svmClassifier)  