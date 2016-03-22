# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年3月14日
time: 下午7:33:07
Function: 
'''
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.svm import SVC 
import numpy as np
from data_pickle import * 
from data_optimize import *
from sklearn import metrics  

def readUIdata(path,attr):
    '''
    path:数据的存储路径
    attribute： 要预测的属性
    '''
    dataset = pickle.load(open(path,"rb"))
    data = delFeatures(dataset.data,attr)
    target = dataset.target
    return data,target
if __name__ == '__main__':
    print 'begain:'
    path = "E:/Desktop/Image/SVMData/sina/data/user_img_pickle.data"
    test_classifiers = ['KNN','LR','RF','DT','GBDT', 'SVM']
    classifiers = {'KNN':KNeighborsClassifier(),  
                   'LR':LogisticRegression(penalty='l2'),  
                   'RF':RandomForestClassifier(n_estimators=8),  
                   'DT':tree.DecisionTreeClassifier(),  
                   'SVM':SVC(kernel='linear', probability=True) ,   
                   'GBDT':GradientBoostingClassifier(n_estimators=200)  
    } 
    target_names={'gender':(['女','男'],[48]),
                  'term':(['网页','手机'],[46]),
                  'edu':(['大学','其他'],[49]),
                  'act_habit_2':(['上午','下午'],[45]),
                  'act_habit_3':(['上午','中午','下午'],[45]),
                  'act_level':(['不活跃','活跃','特别活跃'],[57,56,55]),
                  'act_level_two':(['不活跃','活跃'],[57,56,55]),
                  'influence':(['影响小','一般','影响大'],[59,58,57,56,55,54,44,43,42,]),
                  'influence_two':(['影响小','影响大'],[59,58,57,56,55,54,44,43,42,])}
    
    for predict_lable in target_names.keys():
        #predict_lable = 'influence'#要预测的标签
        data,target = readUIdata(path,target_names[predict_lable][1])
        X = data
        x=[]
        for i in range(len(data.T[1])):
            x.append([data.T[1][i]])
        #print x
        Y = np.array(target[predict_lable])
        print '\n',predict_lable
        #对特征进行归一化处理
        #X = preprocessing.scale(X)
        X = preprocessing.normalize(x)
        #train_x, test_x, train_y, test_y =  train_test_split(X, Y,test_size=0.1)  
        score={
            'precision':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[]},
            'recall':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[]},
            'f1score':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[]},
            'support':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[]},
            'accuracy':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[]}}
        cv = StratifiedKFold(Y,3)
        for train,test in cv:
            #print("%s %s"%(train,test))
            train_x, test_x, train_y, test_y = X[train],X[test], Y[train],Y[test]
            label_class = np.unique(Y) #标签的类别
            print '\n******************* Data Info *********************'  
            print "train data shape:",train_x.shape,"\ntest  data shape:",test_x.shape
              
            for classifier in test_classifiers:  
                #print '\n******************* %s :' % classifiers[classifier].__name__  #打印机器学习方法
                start_time = time.time()  
                model = classifiers[classifier].fit(train_x, train_y)
                #print 'training took %fs!' % (time.time() - start_time)  
                predict = model.predict(test_x)               
                p, r, f1, s = metrics.precision_recall_fscore_support(test_y, predict,average=None)
                a = metrics.accuracy_score(test_y, predict)  
                score['precision'][classifier].append(p.tolist())
                score['recall'][classifier].append(r.tolist())
                score['f1score'][classifier].append(f1.tolist())
                score['support'][classifier].append(s)
                score['accuracy'][classifier].append(a.tolist())
        for classifier in test_classifiers:
            print classifier
            print 'precision:',map(np.mean,np.array(score['precision'][classifier]).T[:])
            print 'recall:',map(np.mean,np.array(score['recall'][classifier]).T[:])
            print 'f1score:',map(np.mean,np.array(score['f1score'][classifier]).T[:])
            print 'accuracy:', mean(score['accuracy'][classifier])
            
    print "\n******************* End! ******************"
 
 