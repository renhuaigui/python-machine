# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
@Created on: 2015年9月11日
@time: 9:38:38
@Function: 
'''

from sklearn import svm
import sklearn.metrics as metrics
from sklearn.cross_validation import train_test_split
import numpy as np
from svm.analysis import analresult
import os
from svm.read import readData
from svm import calculator
from sklearn.feature_extraction.text import  TfidfVectorizer

def SVMtest(base):
    #read data
    X,y=readData(base)
    #tfidf
    #X = TF_IDF(X,y,base)   #这行后面修改
    #SVM with 'rbf'
    kind = 'rbf'
    clf = svm.SVC(kernel = kind)
    #SVM with linear
    lin_clf = svm.LinearSVC()
    x_train, x_test, y_train, y_test = train_test_split(X, y)#(X, y, test_size=0.5, random_state=51)
#     print(y_test)
    
    clf.fit(x_train,y_train)
    preResultlist = clf.predict(x_test)
    svm_score = calculator(y_test,preResultlist)
#     metrics,rate = analresult(y_test,preResultlist)
  
    lin_clf.fit(x_train,y_train)
    prelist = lin_clf.predict(x_test)
    Linear_score = calculator(y_test, prelist)
    return svm_score, Linear_score

def p_result(score):
    for value in score:
        print value 

def result_write(path,data):
    f = open(path,'a') #追加写
    for li in data:
        f.write(str(li))
        f.write("\n")
    f.write("\n")
    f.close()
    
    
svm_score,Linear_score = SVMtest("E:/Desktop/Image/SVMData/gender_wechat.txt")
write_path = "E:/Desktop/Image/SVMData/gender_wechat_result.txt"

result_write(write_path, svm_score)
result_write(write_path, Linear_score)
p_result(svm_score)
print "\n"
p_result(Linear_score)
