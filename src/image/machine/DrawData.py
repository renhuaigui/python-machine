# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年4月21日
time: 上午9:04:15
Function: 
'''
from pylab import *
import cPickle as pickle
from img_predict import *
from src.image.dataProcess.dataset import *
class drawfeatur(object):
    data=None
    def __init__(self):
        '''
        '''
    def setdata(self,data):
        self.data=data
    def loaddataset(self,path):
        return pickle.load(open(path,'rb'))
    def dumpdataset(self,path):
        pickle.dump(self, open(path,'wb'), True)
def drawThreshold():
    datapath ="E:/Desktop/Image/SVMData/sina/data/dataset.data"
    reultpaht ="E:/Desktop/Image/SVMData/sina/data/userinfluecethreshold.data"
    dt =  dataset().loaddataset(datapath)
    classifiers = {'NB':naive_bayes_classifier,   
                  'KNN':knn_classifier,  
                   'LR':logistic_regression_classifier,  
                   'RF':random_forest_classifier,  
                   'DT':decision_tree_classifier,  
                  'SVM':svm_classifier,  
                'SVMCV':svm_cross_validation,  
                 'GBDT':gradient_boosting_classifier  
    }  
    result={'t':[],'p':[],'r':[],'f1':[],'a':[],'f10':[],'f11':[],'p1':[],'p0':[],'r1':[],'r0':[]}
    classifier = 'GBDT'
    X = dt.SiftVisualWord
    userdata = dt.userData
    pImpact = userdata.T[21]
    Y = np.copy(pImpact)
    num = 0.0001
    while(num<=0.02):
        score ={'p':[],'r':[],'f1':[],'a':[]}
        print num
        for i in range(len(pImpact)):
            if pImpact[i]<=num:
                Y[i] = 0
            else:
                Y[i] = 1
        
        print Y.tolist().count(0),Y.tolist().count(1),

        
        print Y
        cv = StratifiedKFold(Y,10)
        for train,test in cv:
            #print("%s %s"%(train,test))
            train_x, test_x, train_y, test_y = X[train],X[test], Y[train],Y[test]
            model = classifiers[classifier](train_x, train_y)
            predict = model.predict(test_x) 
            p, r, f1, s = metrics.precision_recall_fscore_support(test_y, predict,average=None)
            a = metrics.accuracy_score(test_y, predict)  
            score['p'].append(p.tolist())
            score['r'].append(r.tolist())
            score['f1'].append(f1.tolist())
            score['a'].append(a)
            print s
#             print p,r,f1,a,mean(f1)
        print mean(score['p']),mean(score['r']),mean(score['f1'])
        result['t'].append(num)
        result['p'].append(mean(score['p']))
        result['r'].append(mean(score['r']))
        result['f1'].append(mean(score['f1']))
        result['a'].append(mean(score['a']))
        result['f10'].append(mean(np.array(score['f1']).T[0]))
        result['f11'].append(mean(np.array(score['f1']).T[1]))
        result['p0'].append(mean(np.array(score['p']).T[0]))
        result['p1'].append(mean(np.array(score['p']).T[1]))
        result['r0'].append(mean(np.array(score['r']).T[0]))
        result['r1'].append(mean(np.array(score['r']).T[1]))
#         print score['a']
        if num<0.002:
            num = num + 0.00005
        elif num>=0.002 and num<0.01:
            num = num + 0.0001
        else:
            num = num+0.01    
    plt.xlim(0,0.002)
    plt.scatter(result['t'],result['p'],marker = 'x', color = 'm',label = 'pression')
    plt.scatter(result['t'],result['r'], marker = '*', color = 'b', label= 'recall')   
    plt.scatter(result['t'],result['f1'],marker = 'o', color = 'g', label = 'f1-score')     
    plt.scatter(result['t'],result['a'],marker = '+', color = 'r',label = 'accuracy')
    plt.plot(result['t'],result['f10'])
    plt.plot(result['t'],result['f11'])
    plt.legend()
    savefig("E:/Desktop/Image/SVMData/sina/result/treshold.eps")
    plt.show()
   
    rs = drawfeatur()
    rs.setdata(result)
    rs.dumpdataset("E:/Desktop/Image/SVMData/sina/data/treshold.eps.data")
  
  
def drawActivity():
    datapath ="E:/Desktop/Image/SVMData/sina/data/dataset.data"
    reultpaht ="E:/Desktop/Image/SVMData/sina/data/userinfluecethreshold.data"
    dt =  dataset().loaddataset(datapath)
    classifiers = {'NB':naive_bayes_classifier,   
                  'KNN':knn_classifier,  
                   'LR':logistic_regression_classifier,  
                   'RF':random_forest_classifier,  
                   'DT':decision_tree_classifier,  
                  'SVM':svm_classifier,  
                'SVMCV':svm_cross_validation,  
                 'GBDT':gradient_boosting_classifier  
    }  
    result={'t':[],'p':[],'r':[],'f1':[],'a':[],'f10':[],'f11':[],'p1':[],'p0':[],'r1':[],'r0':[]}
    classifier = 'GBDT'
    X = dt.SiftVisualWord
    userdata = dt.userData
    pImpact = userdata.T[14]
    Y = np.copy(pImpact)
    num = 0.0001
    while(num<=0.002):
        score ={'p':[],'r':[],'f1':[],'a':[]}
        print num
        for i in range(len(pImpact)):
            if pImpact[i]<=num:
                Y[i] = 0
            else:
                Y[i] = 1
        
        print Y.tolist().count(1),Y.tolist().count(0),

        
        print Y
        cv = StratifiedKFold(Y,10)
        for train,test in cv:
            #print("%s %s"%(train,test))
            train_x, test_x, train_y, test_y = X[train],X[test], Y[train],Y[test]
            model = classifiers[classifier](train_x, train_y)
            predict = model.predict(test_x) 
            p, r, f1, s = metrics.precision_recall_fscore_support(test_y, predict,average=None)
            a = metrics.accuracy_score(test_y, predict)  
            score['p'].append(p.tolist())
            score['r'].append(r.tolist())
            score['f1'].append(f1.tolist())
            score['a'].append(a)
            #print s
#             print p,r,f1,a,mean(f1)
        print mean(score['p']),mean(score['r']),mean(score['f1'])
        result['t'].append(num)
        result['p'].append(mean(score['p']))
        result['r'].append(mean(score['r']))
        result['f1'].append(mean(score['f1']))
        result['a'].append(mean(score['a']))
        result['f10'].append(mean(np.array(score['f1']).T[0]))
        result['f11'].append(mean(np.array(score['f1']).T[1]))
        result['p0'].append(mean(np.array(score['p']).T[0]))
        result['p1'].append(mean(np.array(score['p']).T[1]))
        result['r0'].append(mean(np.array(score['r']).T[0]))
        result['r1'].append(mean(np.array(score['r']).T[1]))
#         print score['a']
        if num<0.002:
            num = num + 0.00005
        elif num>=0.002 and num<0.01:
            num = num + 0.0001
        else:
            num = num+0.01    
    plt.xlim(0,600)
    plt.scatter(result['t'],result['p'],marker = 'x', color = 'm',label = 'pression')
    plt.scatter(result['t'],result['r'], marker = '*', color = 'b', label= 'recall')   
    plt.scatter(result['t'],result['f1'],marker = 'o', color = 'g', label = 'f1-score')     
    plt.scatter(result['t'],result['a'],marker = '+', color = 'r',label = 'accuracy')
    plt.plot(result['t'],result['f10'])
    plt.plot(result['t'],result['f11'])
    plt.legend()
    savefig("E:/Desktop/Image/SVMData/sina/result/activity.eps")
    plt.show()
   
    rs = drawfeatur()
    rs.setdata(result)
    rs.dumpdataset("E:/Desktop/Image/SVMData/sina/data/activity.eps.data")
      
if __name__ == '__main__':
    drawThreshold()
    