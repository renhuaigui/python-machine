# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2015年10月23日
time: 下午3:10:16
Function: 
'''
import numpy as np     
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report   
from sklearn import svm  
from sklearn import tree  , neighbors
from sklearn.cross_validation import train_test_split  
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from __builtin__ import file

outpath = "E:/Desktop/mechine_result.txt"
def model(x_train, x_test, y_train, y_test,clf):
    print(clf)  
    clf.fit(x_train, y_train)  
    f = file(outpath,'a+')
    f.write(str(clf))
    f.write("\n")
    ''''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''  
#     print(clf.feature_importances_)  
  
    '''''测试结果的打印'''  
    answer = clf.predict(x_test)  
#     print(x_train)  
#     print(answer)  
#     print(y_train)  
    avrage = np.mean(answer == y_test)
    f.write(str(avrage))
    f.write("\n")
    '''''准确率与召回率'''  
    precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))  
#     print (precision)
#     answer = clf.predict_proba(x_trian)[:,1]  
#     print answer
#     answer = answer > 0.3
    report = classification_report(y_test, answer, target_names = ['女', '男'])
    print report
    f.write(str(report))
    f.write("\n\n")
    f.close()

def model_test(x_train, x_test, y_train, y_test):
    ''''' 使用信息熵作为划分标准，对决策树进行训练 '''  
    tree_clf = tree.DecisionTreeClassifier(criterion='entropy')
#     model(x_train, x_test, y_train, y_test, tree_clf)


    ''''' 训练KNN分类器 '''  
    knn_clf = neighbors.KNeighborsClassifier(algorithm='kd_tree') 
   # model(x_train, x_test, y_train, y_test, knn_clf)


    '''' 调用MultinomialNB分类器  '''
    NB_clf = MultinomialNB()
#     model(x_train, x_test, y_train, y_test, NB_clf)


    ''' 逻辑回归分类器'''
    log_clf = LogisticRegression()
   # model(x_train, x_test, y_train, y_test, log_clf)

    ''' 调用SVM分类器'''
    svm_linear_clf = svm.SVC(kernel='linear') 
    svm_poly_clf   = svm.SVC(kernel='poly', degree=3) 
    svm_rbf_clf    = svm.SVC() 
    svm_sig_clf    = svm.SVC(kernel='sigmoid') 
    model(x_train, x_test, y_train, y_test, svm_linear_clf)
#     model(x_train, x_test, y_train, y_test, svm_poly_clf)
#     model(x_train, x_test, y_train, y_test, svm_sig_clf)
    model(x_train, x_test, y_train, y_test, svm_rbf_clf)


