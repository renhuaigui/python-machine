# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年3月11日
time: 下午1:30:25
Function: 
'''
import cPickle as pickle
from sklearn.cluster import KMeans
import os
import numpy as np

class sift_cluster(object):
    def __init__(self,data):
        '''
        data:是一个字典，其中key是uid, value:sift聚类的结果
        '''
        self.data = data

def get_file_list(path):
    #获取文件夹下所有文件（包括子文件）
    current_files = os.listdir(path)
    all_files = []
    for file_name in current_files:
        full_file_name = os.path.join(path, file_name)
        all_files.append(full_file_name)
 
        if os.path.isdir(full_file_name):
            next_level_files = get_file_list(full_file_name)
            all_files.extend(next_level_files)
 
    return all_files

def get_files(path):
    #获路径下的所有文件夹
    return os.listdir(path)
def dataKmeans(data,k):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(data)
    return kmeans
def readData(path):
    data= []
    rf = open(path)
    for line in rf.readlines():
        tokens = line.strip().split(',')
        #print len(tokens)
        #print tokens
        data.append(map(float,tokens))
    #print len(data)
    return np.array(data)

def errorfind():
    files = get_file_list('E:/Desktop/1140435030')
    
    print len(files)
    data =[]
    for file in files:
        print file
        if len(data)==0:
            data = readData(file)
        else:            
            data = np.vstack((data,readData(file)))
        print len(data)
    #print len(data)
def write_sift(data,path):
    wf = open(path,'w')
    for i in range(len(data)):
        for d in data[i]:
            if d==data[i][-1]:
                wf.write(str(d)+'\n')
            else:
                wf.write(str(d)+'\n')
    wf.close()
if __name__ == '__main__':
    print 'begain'
    siftpath = 'E:/SinaSIFT/'
    k=10
    k5=5
    k2=2
    clusterPath = 'E:/Desktop/Image/SVMData/sina/data/sift_cluster('+str(k)+')_data'
    dataset = {}
    user =  get_files(siftpath)
    rf=open("E:/Desktop/userhascluster.txt",'r')
    user_has_cluster = rf.readline().strip().split(',')
    print user_has_cluster
    for u in user_has_cluster:
        print u
        user.remove(u)
    print user
    for i in range(len(user)):
       
        print siftpath+user[i],i
        files = get_file_list(siftpath+user[i])
        data =[]
        print len(files),'file reading data..'
        for file in files:
            data = readData(file)
            if len(data)==0:
                data = readData(file)
            else:            
                data = np.vstack((data,readData(file)))
            #print len(data)
        print 'kmeans train..'
        clfk = dataKmeans(data,k)
        #print clfk.cluster_centers_
        print k,'write to file'
        write_sift(clfk.cluster_centers_, 'D:/siftcluster/10/'+user[i]+'.txt')
        
        clfk5 = dataKmeans(data,k5)
        #print clfk5.cluster_centers_
        print k5,'write to file'
        write_sift(clfk5.cluster_centers_, 'D:/siftcluster/5/'+user[i]+'.txt')
        
        clfk2 = dataKmeans(data,k2)
        #print clfk2.cluster_centers_
        print k2,'write to file'
        write_sift(clfk2.cluster_centers_, 'D:/siftcluster/2/'+user[i]+'.txt')
        #print clfk
        dataset[user[i]] = clfk.cluster_centers_
        wf=open("E:/Desktop/userhascluster.txt",'a')
        wf.write(','+user[i])
        wf.close()
        user_has_cluster.append(user[i])
   
    
    print dataset
    
    siftdata = sift_cluster(dataset)
    pickle.dump(siftdata, open(clusterPath,'wb'),True)
    print pickle.load(open(clusterPath,'rb')).data
    print '\n----------------------'
    print '---------End----------'
    print '----------------------' 

    
    