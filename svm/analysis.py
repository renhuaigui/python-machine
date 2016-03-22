# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on: 2015年9月11日
time :9:54:28
Function: 
'''

import copy
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
def resultAnalysis(resultFilePath):
    """
    count how many satisfied the followed rules:
    real, pred
    (1,1)
    (1,2)
    (1,0)
    (2,0)
    (2,1)
    (2,2)
    """
    resultAnalysis_dict = dict()
    with open(resultFilePath,'r') as fr:
        for eline in fr:
            li = eline.split(',')
            real = li[0]
            pred = li[-1]
            if real == pred:
                if real == '1':
                    resultAnalysis_dict.setdefault('1-1',0)
                    resultAnalysis_dict['1-1'] += 1
                else:
                    resultAnalysis_dict.setdefault('2-2',0)
                    resultAnalysis_dict['2-2'] += 1
            else:
                if real == '1':
                    if pred=='0':
                        resultAnalysis_dict.setdefault('1-0',0)
                        resultAnalysis_dict['1-0'] += 1
                    else:    
                        resultAnalysis_dict.setdefault('1-2',0)
                        resultAnalysis_dict['1-2'] += 1
                else:
                    if pred=='0':
                        resultAnalysis_dict.setdefault('2-0',0)
                        resultAnalysis_dict['2-0'] += 1
                    else:
                        resultAnalysis_dict.setdefault('2-1',0)
                        resultAnalysis_dict['2-1'] += 1 
                    
    return resultAnalysis_dict 

def resultAnalysisValue(resultAnalysis_dict):
    resultAnalysisValueDict = dict()
    p = 1.0*resultAnalysis_dict['1-1']+resultAnalysis_dict['1-2']
    n = 1.0*resultAnalysis_dict['2-1']+resultAnalysis_dict['2-2']
    precision = 1.0*resultAnalysis_dict['1-1']/(resultAnalysis_dict['1-1']+resultAnalysis_dict['2-1'])
    accuracy = (resultAnalysis_dict["1-1"]+resultAnalysis_dict['2-2'])/(p+n)
    recall = resultAnalysis_dict["1-1"]/p
    f1=2*precision*recall/(precision+recall)
    resultAnalysisValueDict.setdefault('precision',precision)
    resultAnalysisValueDict.setdefault('accuracy',accuracy)
    resultAnalysisValueDict.setdefault("recall",recall)
    resultAnalysisValueDict.setdefault("F1",f1)
    return resultAnalysisValueDict

def analresult(y_ture,y_pred):
    classlist = list()
    resultmetrics= list()#用于存储混淆矩阵
    score = list()
    temp =[0]
    #生成一个矩阵[n][n],其中n为y_ture的类数
    for value in y_ture:
        if value not in classlist:
            classlist.append(value)
    for i in range(len(classlist)-1):
        temp.append(0)
    for i in range(len(classlist)):
        nihao = copy.deepcopy(temp)      
        resultmetrics.append(nihao)
    #计数混淆矩阵
    for i,j in zip(y_ture,y_pred):
        resultmetrics[i-1][j-1] += 1
    #用于生成f1,precision,recall,accuracy
    
    score.append(['f1:']+list(f1_score(y_ture,y_pred,average = None)))             
    score.append(['macro:'] + [f1_score(y_ture,y_pred,average = 'macro')])
    score.append(['micro:'] + [f1_score(y_ture,y_pred,average = 'micro')])
    score.append(['weighted:'] + [f1_score(y_ture,y_pred,average = 'weighted')])
    
    score.append(['precision:']+list(precision_score(y_ture,y_pred,average = None)))
    score.append(['macro:'] + [precision_score(y_ture,y_pred,average = 'macro')])
    score.append(['micro:'] + [precision_score(y_ture,y_pred,average = 'micro')])
    score.append(['weighted:'] + [precision_score(y_ture,y_pred,average = 'weighted')])
    
    score.append(['recall:']+list(recall_score(y_ture,y_pred,average = None)))
    score.append(['macro:'] + [recall_score(y_ture,y_pred,average = 'macro')])
    score.append(['micro:'] + [recall_score(y_ture,y_pred,average = 'micro')])
    score.append(['weighted:'] + [recall_score(y_ture,y_pred,average = 'weighted')])
    
    score.append(['accuracy']+[accuracy_score(y_ture,y_pred)])
    return resultmetrics,score
def analresult_for_svdcf(y_ture,y_pred):
    classlist = list()
    resultmetrics= list()#用于存储混淆矩阵
    score = list()
    temp =[0]
    #生成一个矩阵[n][n],其中n为y_ture的类数
    for value in y_ture:
        if value not in classlist:
            classlist.append(value)
    for i in range(len(classlist)-1):
        temp.append(0)
    for i in range(len(classlist)):
        nihao = copy.deepcopy(temp)      
        resultmetrics.append(nihao)
    #计数混淆矩阵
    for i,j in zip(y_ture,y_pred):
        resultmetrics[int(i-1)][int(j-1)] += 1
#         resultmetrics[i-1][j-1] += 1
    #用于生成f1,precision,recall,accuracy
    
    temp = f1_score(y_ture,y_pred,average = None)
    score.append(['f1:']+list(temp)+[np.array(temp).mean()])
    score.append(['precision:']+list(precision_score(y_ture,y_pred,average = None)))
    score.append(['recall:']+list(recall_score(y_ture,y_pred,average = None)))
    score.append(['accuracy']+[accuracy_score(y_ture,y_pred)])
    return resultmetrics,score
            