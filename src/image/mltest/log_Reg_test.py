# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2015年10月23日
time: 下午5:16:35
Function: 
'''

from readData import readData
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from numpy import average
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report  
import time  
#计算运行时间 
start_time = time.time() 


path = "E:/Desktop/Image/SVMData/gender_wechat_scale.txt"
x,y = readData(path)

average = 0
testNum = 10
clf = LogisticRegression()  
print clf
for i in range(0,testNum):
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    clf = LogisticRegression()   
    clf.fit(x_train, y_train)  
    y_pred = clf.predict(x_test)  
    p = np.mean(y_pred == y_test)  
    print(p)  
    average += p  


answer = clf.predict_proba(x_test)[:,1]  
precision, recall, thresholds = precision_recall_curve(y_test, answer)      
report = answer > 0.5  
print(classification_report(y_test, report, target_names = ['neg', 'pos']))  
print("average precision:", average/testNum)  
print("time spent:", time.time() - start_time)  

