# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2015年10月23日
time: 下午2:01:41
Function: 
'''
import numpy as np  
import scipy as sp  
from sklearn import tree  
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report  
from sklearn.cross_validation import train_test_split  
from readData import readData

  
''''' 数据读入 '''  
path = "E:/Desktop/Image/SVMData/loc_train.txt"
x,y = readData(path)


''''x_train 训练数据, 
    x_test  训练标签, 
    y_train 测试数据, 
    y_test  测试标签'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)  
  
''''' 使用信息熵作为划分标准，对决策树进行训练 '''  
clf = tree.DecisionTreeClassifier(criterion='entropy')  


print(clf)  
clf.fit(x_train, y_train)  
  
''''' 把决策树结构写入文件 '''  
with open("tree.dot", 'w') as f:  
    f = tree.export_graphviz(clf, out_file=f)  
      
''''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''  
print(clf.feature_importances_)  
  

'''''测试结果的打印'''  
answer = clf.predict(x_test)  
print(x_train)  
print(answer)  
print(y_train)  
print(np.mean( answer == y_test))  

'''''准确率与召回率'''  
precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))  
answer = clf.predict_proba(x_test)[:,1]  
print(classification_report(y_test, answer, target_names = ['thin', 'fat']))  
