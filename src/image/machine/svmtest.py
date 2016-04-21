
# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年4月1日
time: 下午12:54:27
Function: 
'''
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from img_predict import *
from sklearn.ensemble import GradientBoostingClassifier  


def svmTest(train,target):
    model = SVC(kernel='rbf', probability=True)  
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    C_range = 10.0 ** np.arange(-6,9)
    gamma_range = 10.0 ** np.arange(-6,5)
    param_grid = dict(gamma = gamma_range,C = C_range)
    cv = StratifiedKFold(y=target,n_folds=3)
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv)
    grid.fit(train, target)
    best_parameters = grid.best_estimator_.get_params()  
    for para, val in best_parameters.items():  
        print para, val  
    print("The best classifier is: ", grid.best_estimator_)
    print grid.grid_scores_
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'],probability=True)
    model.fit(train, target)  
    return model,best_parameters
    
    
if __name__ == '__main__':
    path = "E:/Desktop/Image/SVMData/sina/data/user_img_sift.data"
    dataset = pickle.load(open(path,'rb'))
    X = dataset.data.T[:43].T
    X = dataset.siftdata
    Y = np.array(dataset.target['edu'])
    model,best_parameters = svmTest(X,Y)
 
#     gdbt = GradientBoostingClassifier(n_estimators=200)  
#     gdbt.fit(X, Y)
#     gdbt.predict(X)   
#     class_weight = gdbt.feature_importances_   
#     svm = SVC(kernel='rbf', C=10000.0, gamma=0.001,class_weight = 'auto')
#     cv = StratifiedKFold(Y,5)
#     for train,test in cv:
#         #print train
#         train_x, test_x, train_y, test_y = X[train],X[test], Y[train],Y[test]    
#         svm.fit(train_x,train_y)
#         predict = svm.predict(test_x)
#         print  '\n clasification report:\n', metrics.classification_report(test_y, predict)
#         print '\n confussion matrix:\n',metrics.confusion_matrix(test_y, predict)
    
    
    print "\n******************* End! ******************"