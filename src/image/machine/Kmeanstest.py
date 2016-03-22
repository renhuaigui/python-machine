# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年3月8日
time: 下午10:21:00
Function: 
'''
from data_pickle import *
from sklearn.cluster import KMeans


def dataKmeans(data,k):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(data)
    return kmeans
if __name__ == '__main__':
    path = "E:/Desktop/Image/SVMData/sina/data/user_img_pickle.data"
    dataset = pickle.load(open(path,"rb"))
    kclf = dataKmeans(dataset.data, 7)
    print kclf.cluster_centers_
    print kclf.labels_
    for tk in kclf.labels_:
        print tk
    print kclf.inertia_
    
    
    
    