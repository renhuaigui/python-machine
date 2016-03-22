# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2015年10月23日
time: 下午2:47:48
Function:不同的算法进行分类比较
'''
from model import model_test
import numpy as np  
import scipy as sp  
from sklearn import svm  
from sklearn import tree  , neighbors
from sklearn.cross_validation import train_test_split  
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.datasets import load_svmlight_file

if __name__ == '__main__':
    
    path = "E:/Desktop/Image/SVMData/sina/comfrom_libsvmdata.txt"
    x,y = load_svmlight_file(path)
    
    ''''x_train 训练数据, 
        x_test  测试数据, 
        y_train 训练标签,
        y_test  测试标签'''
    x_train, x_test, y_train, y_test = train_test_split(x, y) 
    model_test(x_train, x_test, y_train, y_test)
    
    #把数据划分为10分 用于做交叉验证  
    cv = StratifiedKFold(y,5)
    
    for train,test in cv:
    #     print("%s %s"%(train,test))
        x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
        model_test(x_train, x_test, y_train, y_test)
    print ("End!")
    print x_train
    # count_vec = TfidfVectorizer(binary = False, decode_error = 'ignore',stop_words = 'english')  
    # x_train = count_vec.fit_transform(x_train)  
    # x_test  = count_vec.transform(x_test)  




