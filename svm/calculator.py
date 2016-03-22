# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2015年10月23日
time: 上午11:03:42
Function: 
'''

import copy
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

def calculator(y_ture,y_pred):
    score = list()

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
    return score