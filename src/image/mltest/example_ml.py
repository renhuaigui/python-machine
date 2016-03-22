# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年2月27日
time: 下午2:15:43
Function: 
'''
import numpy as np  
import scipy as sp 
from sklearn.datasets import load_svmlight_file

def Gaussian_Naive_Bayes(data,target):
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    pred = gnb.fit(data, target).predict(data)
    print("Number of mislabled points out of a total %d point :%d "\
          %(len(data),(target != pred).sum()))
    

if __name__ == '__main__':
    path = "E:/Desktop/Image/SVMData/wechat/wang_gender_libsvmdata.txt"
    data,target = load_svmlight_file(path)
    data = data.toarray()
    print type(data)
    Gaussian_Naive_Bayes(data,target)




















