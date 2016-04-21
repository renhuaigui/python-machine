# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年3月30日
time: 下午3:10:04
Function: 
'''
from img_predict import *
from sklearn.ensemble import GradientBoostingClassifier

def predictresult(dataset):
    all_classifiers = ['KNN', 'LR', 'RF', 'DT', 'SVM', 'GBT'] 
    test_classifiers = ['KNN','LR','RF','DT','GBDT']
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
    target_names={'gender':(['女','男'],[6]),
                  'term':(['网页','手机'],[4]),
                  'edu':(['大学','其他'],[7]),
                  'act_habit_2':(['上午','下午'],[3]),
                  'act_habit_3':(['上午','中午','下午'],[3]),
                  'act_level':(['不活跃','活跃','特别活跃'],[14,13,12]),#,56,55
                  'act_level_two':(['不活跃','活跃'],[14,13,12]),
                  'influence':(['影响小','一般','影响大'],[17,16,15,14,13,12,11,10,9,8,2,1,0]),#,58,57,56,55,54,44,43,42,
                  'influence_two':(['影响小','影响大'],[17,16,15,14,13,12,11,10,9,8,2,1,0]),
                  'user_influence':(['影响小','影响大'],[17,16,15,14,13,12,11,10,9,8]),
                  'f_influence':(['影响小','影响大'],[17,16,15,14,13,12,11,10,9,8,2,1,0]),
                  }
    target_name = ['gender','term','edu','act_habit_2','act_habit_3','act_level','act_level_two','influence','influence_two','user_influence','f_influence']
#     target_name = ['f_influence','user_influence']
    predict_label = target_name[10]
    print predict_label
    print 'reading training and testing data...'  
    imgdata = dataset.data
    target = dataset.target
    siftdata = dataset.siftdata
    userdata = imgdata.T[42:60].T
    userdata = delFeatures(userdata,target_names[predict_label][1])    
    delline = [63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42]
    imgdata = delFeatures(imgdata, delline)
    Y = np.array(target[predict_label])
    print imgdata.shape,siftdata.shape,userdata.shape
    X1 = userdata
    X3 = np.column_stack((X1,GradientBoostingClassifier(n_estimators=200).fit_transform(imgdata,Y)))
    X4 = np.column_stack((X3,GradientBoostingClassifier(n_estimators=200).fit_transform(siftdata,Y)))
    X2 = np.column_stack((GradientBoostingClassifier(n_estimators=200).fit_transform(imgdata,Y),GradientBoostingClassifier(n_estimators=200).fit_transform(siftdata,Y)))
    X1 = userdata
    X2 = np.column_stack((imgdata,siftdata))
    X3 = np.column_stack((userdata,imgdata))
    X4 = np.column_stack((X3,siftdata))
    X_T = [X1,X2,X3,X4]
    resultclass =[1,2,3,4]
    
    result = {'NB':[[],[],[],[],[],[],[]],   
             'KNN':[[],[],[],[],[],[],[]],  
              'LR':[[],[],[],[],[],[],[]],  
              'RF':[[],[],[],[],[],[],[]],  
              'DT':[[],[],[],[],[],[],[]],  
             'SVM':[[],[],[],[],[],[],[]],  
           'SVMCV':[[],[],[],[],[],[],[]],  
            'GBDT':[[],[],[],[],[],[],[]]  
    }

    score={
            'precision':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]},
            'recall':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]},
            'f1score':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]},
            'support':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]},
            'accuracy':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]},
            'predict_result':{'NB':[],'KNN':[],'LR':[],'RF':[], 'DT':[], 'SVM':[], 'GBDT':[],'SVMCV':[]}
            }
#     X=  GradientBoostingClassifier(n_estimators=200).fit_transform(X,Y)
#     X = siftdata

    for X in X_T:
        print X.shape,Y.shape
        cv = StratifiedKFold(Y,10)
    #     X = preprocessing.normalize(X)
        for train,test in cv:
             #print train
            train_x, test_x, train_y, test_y = X[train],X[test], Y[train],Y[test]
            uid = np.array(target['uid'])[test]
            label_class = np.unique(Y) #标签的类别
            for classifier in test_classifiers:  
                #print '\n******************* %s :' % classifiers[classifier].__name__  #打印机器学习方法
                model = classifiers[classifier](train_x, train_y)  
                predict = model.predict(test_x) 
                    
                p, r, f1, s = metrics.precision_recall_fscore_support(test_y, predict,average=None)
    
                a = metrics.accuracy_score(test_y, predict)  
                score['precision'][classifier].append(p.tolist())
                score['recall'][classifier].append(r.tolist())
                score['f1score'][classifier].append(f1.tolist())
                score['support'][classifier].append(s)
                score['accuracy'][classifier].append(a.tolist())
                score['predict_result'][classifier] = class_reult(test_y, predict, uid, score['predict_result'][classifier])
        for classifier in test_classifiers:
            print classifier
            print 'precision:',map(np.mean,np.array(score['precision'][classifier]).T[:]),mean(score['precision'][classifier])
            print 'recall:',map(np.mean,np.array(score['recall'][classifier]).T[:]),mean(score['recall'][classifier])
            print 'f1score:',map(np.mean,np.array(score['f1score'][classifier]).T[:]),mean(score['f1score'][classifier])
            print 'accuracy:', mean(score['accuracy'][classifier])
            print 'support',map(np.mean,np.array(score['support'][classifier]).T[:])
            result[classifier][0].append(mean(np.array(score['precision'][classifier]).T[0]))
            result[classifier][1].append(mean(np.array(score['precision'][classifier]).T[1]))
            result[classifier][2].append(mean(np.array(score['recall'][classifier]).T[0]))
            result[classifier][3].append(mean(np.array(score['recall'][classifier]).T[1]))
            result[classifier][4].append(mean(np.array(score['f1score'][classifier]).T[0]))
            result[classifier][5].append(mean(np.array(score['f1score'][classifier]).T[1]))
            result[classifier][6].append(mean(score['accuracy'][classifier]))    
        print len(score['predict_result']['GBDT'])
    #plt.ylim(0.72,0.9)
    for classifier in test_classifiers: 
        plot(resultclass,result[classifier][6],label='accuracy')
    
        plot(resultclass,result[classifier][0],label='0-precision')
        plot(resultclass,result[classifier][1],label='1-precision')
    
        plot(resultclass,result[classifier][2],label='0-recall')
        plot(resultclass,result[classifier][3],label='1-recall')
    
        plot(resultclass,result[classifier][4],label='0-f1score')
        plot(resultclass,result[classifier][5],label='1-f1score')
        plt.legend(loc=2,ncol=3)
        plt.title(predict_label+classifier)
#         savefig("E:/Desktop/Image/SVMData/sina/result/"+predict_label+classifier+"_result.png")
        show()
        plt.close()
   
 

if __name__ == '__main__':
    path = "E:/Desktop/Image/SVMData/sina/data/user_img_sift.data"
    dataset = pickle.load(open(path,'rb'))
    predictresult(dataset)
    print "\n******************* End! ******************"
    