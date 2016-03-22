# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2015年10月14日
time: 下午8:03:46
Function: 
'''
from sklearn.datasets import load_svmlight_file
from sklearn import metrics 
from sklearn.svm import SVC  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans  
from sklearn.naive_bayes import MultinomialNB 

print("*********   setup 1:  load data     ****************")
#该方法导入的是libsvm格式的数据
X_train, y_train = load_svmlight_file("E:/Desktop/Image/SVMData/loc_train.txt")
X_test,y_test = load_svmlight_file("E:/Desktop/Image/SVMData/loc_predict.txt")

#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=51)
#将数据随机分成测试集和训练集

def calculate_result(actual,pred):  
    print("*********  caclulate precision ,recall,f1 ********** ")
    m_precision = metrics.precision_score(actual,pred);  
    m_recall = metrics.recall_score(actual,pred);  
    print pred
    print 'predict info:'  
    print '  precision:{0:.3f}'.format(m_precision)  
    print '  recall:{0:0.3f}'.format(m_recall);  
    print '  f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred)); 


print ('*********   setup 2: SVM       *********************')  
svclf = SVC(kernel = 'linear')#default with 'rbf'  kernel = 'linear'
svclf.fit(X_train,y_train)   
pred = svclf.predict(X_test);  
calculate_result(y_test,pred);  


print ('*********   setup 3: KNN       *********************') 
knnclf = KNeighborsClassifier()#default with k=5  
knnclf.fit(X_train,y_train)  
pred = knnclf.predict(X_test);  
calculate_result(y_test,pred);
 
#   
# print ('*********   setup 4: KMeans    *********************') 
# pred = KMeans(n_clusters=5)
# pred.fit(X_test)  
# calculate_result(y_test,pred.labels_);
  
  
  
# print ('*********   setup 4: naive_bayes    ****************') 
# clf = MultinomialNB(alpha = 0.01)   
# clf.fit(X_train,y_train);  
# pred = clf.predict(y_test);  
# calculate_result(X_test,pred); 