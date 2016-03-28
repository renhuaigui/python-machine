# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年1月7日
time: 上午11:14:39
Function: 
'''
import sys  
import os  
import time  
from sklearn import metrics  
import numpy as np  
import cPickle as pickle  
from sklearn.datasets import load_svmlight_file  
from sklearn.cross_validation import train_test_split,StratifiedKFold
from sklearn import preprocessing
from data_pickle import * 
from pylab import *


  
# Multinomial Naive Bayes Classifier  
def naive_bayes_classifier(train_x, train_y):  
    from sklearn.naive_bayes import MultinomialNB  
    model = MultinomialNB(alpha=0.01)  
    model.fit(train_x, train_y)  
    return model  
  
  
# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# Logistic Regression Classifier  
def logistic_regression_classifier(train_x, train_y):  
    from sklearn.linear_model import LogisticRegression  
    model = LogisticRegression(penalty='l2')  
    model.fit(train_x, train_y)  
    return model  
  
  
# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=8)  
    model.fit(train_x, train_y)  
    return model  
  
  
# Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):  
    from sklearn import tree  
    model = tree.DecisionTreeClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# GBDT(Gradient Boosting Decision Tree) Classifier  
def gradient_boosting_classifier(train_x, train_y):  
    from sklearn.ensemble import GradientBoostingClassifier  
    model = GradientBoostingClassifier(n_estimators=200)  
    model.fit(train_x, train_y)  
    return model  
  
  
# SVM Classifier  
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf')  
    model.fit(train_x, train_y)  
    return model  
  
# SVM Classifier using cross validation  
def svm_cross_validation(train_x, train_y):  
    from sklearn.grid_search import GridSearchCV  
    from sklearn.svm import SVC  
   
    '''参数寻优'''
    model = SVC(kernel='rbf', probability=True)  
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)  
    grid_search.fit(train_x, train_y)  
    best_parameters = grid_search.best_estimator_.get_params()  
    for para, val in best_parameters.items():  
        print para, val  
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)  
    model.fit(train_x, train_y)  
    return model  

def cross_validation_test():
    from sklearn.cross_validation import StratifiedKFold, cross_val_score
    from sklearn import tree
    from sklearn.naive_bayes import MultinomialNB 
    from sklearn.neighbors import KNeighborsClassifier 
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier 
    from sklearn.svm import SVC 
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
                  'influence':(['影响小','一般','影响大'],[59,58,57,56,55,54,44,43,42,])}
    predict_lable = 'edu'#要预测的标签
    path = "E:/Desktop/Image/SVMData/sina/data/user_img_pickle.data"
    data,target = readUIdata(path,target_names[predict_lable][1])
    X = data
    Y = target[predict_lable]
    print predict_lable
    #对特征进行归一化处理
    #X = preprocessing.scale(X)
    X = preprocessing.normalize(X)
    print len(X.T[0:2].T),len(Y)
    cv = StratifiedKFold(Y,10)
    for classifier in classifiers.keys():
        print classifier      
        score = cross_val_score(classifiers[classifier],X,Y,cv=cv,scoring='f1')
        print score
        print np.mean(score)
        score = cross_val_score(classifiers[classifier],X,Y,cv=cv,scoring='recall')
        print score
        print np.mean(score)
        score = cross_val_score(classifiers[classifier],X,Y,cv=cv,scoring='precision')
        print score
        print np.mean(score)
        score = cross_val_score(classifiers[classifier],X,Y,cv=cv,scoring='accuracy')
        print score
        print np.mean(score)

def read_data(data_file):  
    data, target = load_svmlight_file(data_file)
    return data,target 

def cut_user(data,target):
    dataset = []
    edulabel = []
    rf = open("E:/Desktop/Image/SVMData/sina/label/userhasedu.txt",'r')
    edu_user = rf.readline().strip().split(',')
    for user in edu_user:
        index = target['uid'].index(int(user))
        dataset.append(data[index])
        edulabel.append(target['edu'][index])
    
    print len(edulabel),len(dataset)
    return np.array(dataset),np.array(edulabel)
def update_user(data,target):
    rf = open("E:/Desktop/Image/SVMData/sina/label/userhasedu.txt",'r')
    edu_user = rf.readline().strip().split(',')
    print edu_user
    for user in target['uid']:
        if str(user) not in edu_user:
            index = target['uid'].index(int(user))
            target['edu'][index] = 2
    print len(data),len(target['edu'])
    print target['edu']
    return data,np.array(target['edu'])

def delFeatures(data,delline):
    """
        把区分度不大的特征删除
    data:要处理的数据
    delline: 要删除的列号
    """
    deldata=data
    print '删除：',delline
    for i in delline:
        deldata = np.delete(deldata, i, 1)#删除第i列的元素值
    return deldata
    

def readUIdata(path,attr):
    '''
    path:数据的存储路径
    attribute： 要预测的属性
    '''
    dataset = pickle.load(open(path,"rb"))
    data = delFeatures(dataset.data,attr)
    target = dataset.target
    return data,target
    
def img_machine_test():
    all_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBT'] 
    test_classifiers = ['KNN','LR','RF','DT', 'SVM','GBDT']
#     test_classifiers = ['GBDT']
    classifiers = {'NB':naive_bayes_classifier,   
                  'KNN':knn_classifier,  
                   'LR':logistic_regression_classifier,  
                   'RF':random_forest_classifier,  
                   'DT':decision_tree_classifier,  
                  'SVM':svm_classifier,  
                'SVMCV':svm_cross_validation,  
                 'GBDT':gradient_boosting_classifier  
    }  
    print 'reading training and testing data...'  
    
    
#     X,Y = read_data(data_file)
#     print type(X),type(Y[0])
#     print Y[:10]
    
#     datapath = "E:/Desktop/Image/SVMData/sina/data/img_pickle.data"#对象持久化文件
#     print datapath
#     dataset = pickle.load(open(datapath,"rb"))
#     X = dataset.data
#     Y = map(float, dataset.target.T[5])
#     #删除区分度不大的特征，删除标签特征
#     #deline = [38,37,36,35,31,30,27,26,23,22,19,18,16,14,12,11,9,0]
#     deline = [46]
#     X = delFeatures(X,deline)

    path = "E:/Desktop/Image/SVMData/sina/data/user_img_pickle1.data"
#     path = "E:/Desktop/Image/SVMData/sina/data/sift2class_pickle.data"
    target_names={'gender':(['女','男'],[48]),
                  'term':(['网页','手机'],[46]),
                  'edu':(['大学','其他'],[49]),
                  'act_habit_2':(['上午','下午'],[45]),
                  'act_habit_3':(['上午','中午','下午'],[45]),
                  'act_level':(['不活跃','活跃','特别活跃'],[57]),#,56,55
                  'act_level_two':(['不活跃','活跃'],[57]),
                  'influence':(['影响小','一般','影响大'],[59]),#,58,57,56,55,54,44,43,42,
                  'influence_two':(['影响小','影响大'],[59]),
                  'user_influence':(['影响小','影响大'],[59]),
                  'f_influence':(['影响小','影响大'],[59]),
                  }
    target_name = ['gender','term','edu','act_habit_2','act_habit_3','act_level','act_level_two','influence','influence_two','user_influence']
    target_name = ['act_habit_2']
    for predict_lable in target_name:
        #predict_lable = 'influence'#要预测的标签
        print 
        data,target = readUIdata(path,target_names[predict_lable][1])
        X = data
        Y = np.array(target[predict_lable])
        if predict_lable =='edu':
            X,Y = cut_user(X,target)
        delline = [41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
#         delline = [40,39,38,36,34,33,32,31,29,28,27,25,24,23,21,20,19,17,16,15,14,12,11,10,8,7,5,4,2,1,0]
        delline = [62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42]
#         delline = [61,60,59,58,57,44,43,42]
#         delline = [61,60,59]
        X = delFeatures(data, delline)
        print predict_lable
        #对特征进行归一化处理
        #X = preprocessing.scale(X)
        X = preprocessing.normalize(X)
        #train_x, test_x, train_y, test_y =  train_test_split(X, Y,test_size=0.1)  
        score={
            'precision':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]},
            'recall':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]},
            'f1score':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]},
            'support':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]},
            'accuracy':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]}
            }
        class_weight = []
        cv = StratifiedKFold(Y,10)
        for train,test in cv:
            #print("%s %s"%(train,test))
            train_x, test_x, train_y, test_y = X[train],X[test], Y[train],Y[test]
            label_class = np.unique(Y) #标签的类别
            #print '\n******************* Data Info *********************'  
            #print "train data shape:",train_x.shape,"\ntest  data shape:",test_x.shape
              
            for classifier in test_classifiers:  
                #print '\n******************* %s :' % classifiers[classifier].__name__  #打印机器学习方法
                start_time = time.time()  
                model = classifiers[classifier](train_x, train_y)  
                #print 'training took %fs!' % (time.time() - start_time)  
                start_time = time.time()
                predict = model.predict(test_x) 
                if classifier =='GBDT':
                    class_weight.append(model.feature_importances_.tolist())   #输出权重
                    #print map(lambda x:x*10,class_weight[i][:])
                #print 'predict took %fs!' % (time.time() - start_t ime)
                #print predict  
                #if model_save_file != None:  
                    #model_save[classifier] = model  
                
                p, r, f1, s = metrics.precision_recall_fscore_support(test_y, predict,average=None)
                a = metrics.accuracy_score(test_y, predict)  
                score['precision'][classifier].append(p.tolist())
                score['recall'][classifier].append(r.tolist())
                score['f1score'][classifier].append(f1.tolist())
                score['support'][classifier].append(s)
                score['accuracy'][classifier].append(a.tolist())
    #             print 'precision',p
    #             print 'recall',r
    #             print 'f1score',f1
    #             print 'support',s
    #             print 'accuracy',a
    #             print(metrics.confusion_matrix(test_y, predict))
        #         print(metrics.classification_report(test_y, predict,target_names=target_names[predict_lable][0]))
        #         print(metrics.confusion_matrix(test_y, predict))
        #         accuracy = metrics.accuracy_score(test_y, predict)  
                #print 'accuracy: %.2f%%' % (100 * accuracy)
               
    #             wf = file('E:/Desktop/Image/SVMData/sina/result/'+classifier+'.txt',"a+") 
    #             for i in range(len(test_y)):
    #                     wf.write(str(test_y[i])+','+str(predict[i])+'\n')
    #             wf.close()
    #     
    #             result_class = {}
    #             for i in range(len(test_y)):
    #                 key = str(test_y[i])+'-->'+str(predict[i])
    #                 if key in result_class:
    #                     result_class[key] +=1
    #                 else :
    #                     result_class[key] = 1
    #             print result_class  
    #             
    #         if model_save_file != None:  
    #             pickle.dump(model_save, open(model_save_file, 'wb'))  
        for classifier in test_classifiers:
            print classifier
            print 'precision:',map(np.mean,np.array(score['precision'][classifier]).T[:]),mean(score['precision'][classifier])
            print 'recall:',map(np.mean,np.array(score['recall'][classifier]).T[:]),mean(score['recall'][classifier])
            print 'f1score:',map(np.mean,np.array(score['f1score'][classifier]).T[:]),mean(score['f1score'][classifier])
            print 'accuracy:', mean(score['accuracy'][classifier])
            print 'support',map(np.mean,np.array(score['support'][classifier]).T[:])
        print 'classweight:',map(np.mean,np.array(class_weight).T[:])
        
def followInfluence():

    test_classifiers = ['GBDT']
    classifiers = {'NB':naive_bayes_classifier,   
                  'KNN':knn_classifier,  
                   'LR':logistic_regression_classifier,  
                   'RF':random_forest_classifier,  
                   'DT':decision_tree_classifier,  
                  'SVM':svm_classifier,  
                'SVMCV':svm_cross_validation,  
                 'GBDT':gradient_boosting_classifier  
    }  
    print 'reading training and testing data...'  
    
    path = "E:/Desktop/Image/SVMData/sina/data/user_img_pickle1.data"
#     path = "E:/Desktop/Image/SVMData/sina/data/sift2class_pickle.data"
    target_names={'gender':(['女','男'],[48]),
                  'term':(['网页','手机'],[46]),
                  'edu':(['大学','其他'],[49]),
                  'act_habit_2':(['上午','下午'],[45]),
                  'act_habit_3':(['上午','中午','下午'],[45]),
                  'act_level':(['不活跃','活跃','特别活跃'],[57]),#,56,55
                  'act_level_two':(['不活跃','活跃'],[57]),
                  'influence':(['影响小','一般','影响大'],[59]),#,58,57,56,55,54,44,43,42,
                  'influence_two':(['影响小','影响大'],[59]),
                  'user_influence':(['影响小','影响大'],[59]),
                  'f_influence':(['影响小','影响大'],[59]),
                  }
    target_name = ['gender','term','edu','act_habit_2','act_habit_3','act_level','act_level_two','influence','influence_two','user_influence']
    target_name = ['f_influence']
    for predict_lable in target_name:
        #predict_lable = 'influence'#要预测的标签
        print 
        data,target = readUIdata(path,target_names[predict_lable][1])
        X = data[:]
        Y = np.array(target[predict_lable])
        if predict_lable =='edu':
            X,Y = cut_user(X,target)
        followinf =[]
        j = 0
        num = 0.0001
        while(num<=0.1):
            
            X = data[:]
            followinf.append([num])
            print num
            for i in range(len(X.T[62])):
                if X.T[62][i]<=num:
                    Y[i] = 0
                else:
                    Y[i] = 1
            delline = [41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
    #         delline = [40,39,38,36,34,33,32,31,29,28,27,25,24,23,21,20,19,17,16,15,14,12,11,10,8,7,5,4,2,1,0]
            delline = [61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42]
            delline = [62,61,60,59,58,57,56,55,54,44,43,42]
    #         delline = [61,60,59]
            X = delFeatures(data, delline)
            print predict_lable
            #对特征进行归一化处理
            #X = preprocessing.scale(X)
            X = preprocessing.normalize(X)
            #train_x, test_x, train_y, test_y =  train_test_split(X, Y,test_size=0.1)  
            score={
                'precision':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]},
                'recall':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]},
                'f1score':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]},
                'support':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]},
                'accuracy':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]}
                }
            class_weight = []
            cv = StratifiedKFold(Y,10)
            for train,test in cv:
                #print("%s %s"%(train,test))
                train_x, test_x, train_y, test_y = X[train],X[test], Y[train],Y[test]
                label_class = np.unique(Y) #标签的类别
                #print '\n******************* Data Info *********************'  
                #print "train data shape:",train_x.shape,"\ntest  data shape:",test_x.shape
                  
                for classifier in test_classifiers:  
                    #print '\n******************* %s :' % classifiers[classifier].__name__  #打印机器学习方法
                    start_time = time.time()  
                    model = classifiers[classifier](train_x, train_y)  
                    #print 'training took %fs!' % (time.time() - start_time)  
                    start_time = time.time()
                    predict = model.predict(test_x) 
#                     if classifier =='GBDT':
#                         class_weight.append(model.feature_importances_.tolist())   #输出权重 
                    
                    p, r, f1, s = metrics.precision_recall_fscore_support(test_y, predict,average=None)
                    a = metrics.accuracy_score(test_y, predict)  
                    score['precision'][classifier].append(p.tolist())
                    score['recall'][classifier].append(r.tolist())
                    score['f1score'][classifier].append(f1.tolist())
                    score['support'][classifier].append(s)
                    score['accuracy'][classifier].append(a.tolist())
            for classifier in test_classifiers:
#                 print classifier
                followinf[j].append(mean(np.array(score['precision'][classifier]).T[0]))
                followinf[j].append(mean(np.array(score['precision'][classifier]).T[1]))
#                 print 'precision:',map(np.mean,np.array(score['precision'][classifier]).T[:]),mean(score['precision'][classifier])
                followinf[j].append(mean(np.array(score['recall'][classifier]).T[0]))
                followinf[j].append(mean(np.array(score['recall'][classifier]).T[1]))
#                 print 'recall:',map(np.mean,np.array(score['recall'][classifier]).T[:]),mean(score['recall'][classifier])
                followinf[j].append(mean(np.array(score['f1score'][classifier]).T[0]))
                followinf[j].append(mean(np.array(score['f1score'][classifier]).T[1]))
#                 print 'f1score:',map(np.mean,np.array(score['f1score'][classifier]).T[:]),mean(score['f1score'][classifier])
                followinf[j].append(mean(score['accuracy'][classifier]))
#                 print 'accuracy:', mean(score['accuracy'][classifier])
#                 print 'support',map(np.mean,np.array(score['support'][classifier]).T[:])
#             print 'classweight:',map(np.mean,np.array(class_weight).T[:])
            print  followinf[j]
            
#             if num<2000:
#                 num = num + 100
#             elif num>=2000 and num<10000:
#                 num = num + 500
#             elif num>100000:
#                 num = num +10000
#             else:
#                 num = num+1000
            if num<0.001:
                num = num + 0.0001
            elif num>=0.001 and num<0.01:
                num = num + 0.001
            else:
                num = num+0.1
            j = j+1
        influenceresult = dataPickle(followinf)
        print followinf
        pickle.dump(influenceresult, open('E:/Desktop/Image/SVMData/sina/data/influenceresult2.data', 'wb'),True)
        datainfluenceresult = np.array(followinf)
        plot(datainfluenceresult.T[0],datainfluenceresult.T[7],label='accuracy')
        plt.legend()
        savefig("E:/Desktop/Image/SVMData/sina/result/inf-accuracy.png")
#         show()
#         plt.close()
        plot(datainfluenceresult.T[0],datainfluenceresult.T[1],label='0-precision')
        plot(datainfluenceresult.T[0],datainfluenceresult.T[2],label='1-precision')
        plt.legend()
        savefig("E:/Desktop/Image/SVMData/sina/result/inf-precision.png")
#         show()
#         plt.close()
        plot(datainfluenceresult.T[0],datainfluenceresult.T[3],label='0-recall')
        plot(datainfluenceresult.T[0],datainfluenceresult.T[4],label='1-recall')
        plt.legend()
        savefig("E:/Desktop/Image/SVMData/sina/result/inf-recall.png")
#         show()
#         plt.close()

        plot(datainfluenceresult.T[0],datainfluenceresult.T[5],label='0-f1score')
        plot(datainfluenceresult.T[0],datainfluenceresult.T[6],label='1-f1score')
        plt.legend()
        savefig("E:/Desktop/Image/SVMData/sina/result/inf-f1score.png")

        show()
        plt.close()

if __name__ == '__main__':  
    
#     followInfluence()
#     cross_validation_test()
    img_machine_test()
    
    
    
    #data_file = "E:/Desktop/Image/SVMData/sina/data/sina_gender_libsvmdata_scale_del.txt"  
    #data_file = "E:/Desktop/Image/SVMData/sina/data/sift_gender_libsvmdata.txt" 
    #data_file =  "E:/Desktop/Image/SVMData/sina/data/comfrom_libsvmdata.txt"
#  
#     model_save_file = None 
#     model_save = {}  

        
    print "\n******************* End! ******************"
    

    
    