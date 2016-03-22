# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2015年10月26日
time: 下午1:34:24
Function: 
'''

from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn import svm


path = "E:/Desktop/Image/SVMData/gender_wechat_scale.txt"
x,y = readData(path)


cv = StratifiedKFold(y,10)

# 划分后的数据
for train,test in cv:
    print("%s %s"%(train,test))
    X_train, X_test, y_train, y_test = x[train], x[test], y[train], y[test]
    
clf = svm.SVC()
# 交叉验证的结果
score = cross_val_score(clf,x,y,cv=cv)
print score


