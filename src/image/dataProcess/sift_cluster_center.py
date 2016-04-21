# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年4月5日
time: 下午7:58:13
Function: 
'''
from src.image.dataProcess.sift_extract import sift_extract
from src.image.dataProcess.dataKMeans import dataKmeans
from src.image.dataProcess.dataKMeans import kMeans
import cPickle as pickle
import numpy as np
from scipy import linalg
import dataset
import data_pickle

class cluster(object):
    centers=None
    visualword = None
    def __init__(self):
        "None"
    def visual(self,centers=None,visualword=None):
        '''
        centers 是类别中心， visualword 是数据在各类别中心的词频统计
        '''
        self.centers = centers
        self.visualword = visualword
    def loaddataset(self,path):
        return pickle.load(open(path,'rb'))
    def dumpdataset(self,path):
        pickle.dump(self, open(path,'wb'), True)
    def calcenter(self,path):
        data =[]
        f = open(path)
        for line in f.readlines():
            datatemp = line.strip().split(',')
            data.append(map(float, datatemp))
        data = np.array(data)
        print len(data)
        self.centers = dataKmeans(50,data).cluster_centers_
        return self.centers
    def calword(self,data):
        m = len(data)
        k = len(self.centers)
        word = np.zeros(k)
        for i in range(m):
            index = -1
            minDist = np.inf
            for j in xrange(k):
                distJI = self.distEclud(self.centers[j,:],data[i,:])
                if distJI < minDist:
                    minDist = distJI; index = j
            word[index] = word[index]+1
        return word
    
    def distEclud(self,vecA,vecB):
        #return linalg.norm(vecA, vecB)
        return np.sqrt(sum(np.power(vecA-vecB,2)))
 
def siftcluster():
    path = "E:/Desktop/Image/SVMData/sina/data/image.txt"
    siftpath = "E:/Desktop/Image/SVMData/sina/data/siftimagemean.txt"
    siftcenterpath = "E:/Desktop/Image/SVMData/sina/data/siftcenter.data"
     
    clust = cluster()
    clustercents = clust.calcenter(path=siftpath)
    pickle.dump(clust, open(siftcenterpath,'wb'),True)
    
    clust = pickle.load(open(siftcenterpath,'rb'))
    print len(clust.centers)
    imgNames =[]
    df = open(path) 
    for line in df.readlines(): 
        datatemp = line.strip().split(',') 
        imgNames.append(datatemp[:2]) 
    df.close()
    print clust.centers
    for i in range(len(imgNames)):    
#     for i in range(10):
        imgname = imgNames[i][0]
        print i,imgname
        siftdata = sift_extract(imgname)
        #print siftdata
        l =len(siftdata)
        word = clust.calword(data = siftdata)
        pword = (50*word)/l
        imgNames[i].append(pword)
    clust.visualword = imgNames
    clust.dumpdataset("E:/Desktop/Image/SVMData/sina/data/siftvisulword.data")
        #print imgNames[i]
    imgNames = np.array(imgNames)
    return  imgNames.T[0],imgNames.T[2:].T

def lowfeaturecluster():
    datapath = "E:/Desktop/Image/SVMData/sina/data/image.txt"
    contentpath = "E:/Desktop/Image/SVMData/sina/label/content.txt"
    data = [] 
    imgNames = []
    content =[]
    uids =[]
    visualword =[]
    cf = open(contentpath)
    for line in cf.readlines():
        datatemp = line.strip().split(',')
        content.append(datatemp)
    df = open(datapath) 
    for line in df.readlines(): 
        datatemp = line.strip().split(',') 
        #print datatemp
        imgNames.append(datatemp[:2]) 
        #rint imgNames
        #print datatemp[2:]
        data.append(map(float, datatemp[2:]))
    df.close()
    print 'read over'
    KMclust = dataKmeans(50,data)
    centers = KMclust.cluster_centers_
    labels =  KMclust.labels_
    inertia = KMclust.inertia_
    content = np.array(content)
    mids = content.T[1].tolist()
    print len(labels)
    
    for i in range(len(imgNames)):
        mid = imgNames[i][1]     
        uid = content[mids.index(mid)][0]
        if uid in uids:
            j = uids.index(uid)
            #print labels[i]
            visualword[j][labels[i]] = visualword[j][labels[i]] + 1
            #print  uids[j],visualword[j] 
        else :
            uids.append(uid)
            #print len(centers)
            a = [0]*len(centers)
            print uid
            a[labels[i]] = a[labels[i]] + 1
            visualword.append(a)
            
    visualword = np.array(visualword)
    clust = cluster()
    clust.centers = uids
    clust.visualword = visualword
    clust.dumpdataset("E:/Desktop/Image/SVMData/sina/data/lowfeaturevisulword.data")
  
    print visualword
    print len(visualword)
    return uids,visualword
    
    
if __name__ == '__main__':
    lowfeaturecluster()
#     clust = pickle.load(open("E:/Desktop/Image/SVMData/sina/data/siftvisulword.data",'rb'))
#     data = dataset.dataset()
#     clust =cluster()
#     clust = clust.loaddataset("E:/Desktop/Image/SVMData/sina/data/siftvisulword.data")
#     dat = data_pickle.dataPickle()
#     sd = dat.loaddataset("E:/Desktop/Image/SVMData/sina/data/img_pickle.data1")
#     print sd.data
#     data.SiftData = clust.visualword
#     data.dumpdataset("E:/Desktop/Image/SVMData/sina/data/siftvisulwordforimg.data")
#     data = dataPickle.loaddataset("E:/Desktop/Image/SVMData/sina/data/img_pickle.Siftdata")
#     print data.userdata
#     print clust.centers
#     print clust.visualword[0]
      
    print '****end!***'