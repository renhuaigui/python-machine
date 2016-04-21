# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年3月1日
time: 下午5:27:35
Function: 
'''
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import linalg
from numpy import *


def dataKmeans(k,data):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(data)
    return kmeans

def distCos(vecA,vecB):
    return dot(vecA,vecB)/(linalg.norm(vecA)*linalg.norm(vecB))
def distEclud(vecA,vecB):
    #return linalg.norm(vecA, vecB)
    return sqrt(sum(power(vecA-vecB,2)))
def randcenters(data,k):
    n = shape(data)[1]
    clustercenters = mat(zeros((k,n)))
    for col in xrange(n):
        mincol  = min(data[:,col])
        rangeJ = float(max(array(data)[:,col]) - mincol)
        clustercenters[:,col] = mincol + rangeJ * random.rand(k,1)
    return clustercenters
    
def kMeans(dataSet,clustercents,k,distMeas=distEclud,iter = 300):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points to a centroid, also holds SE of each point
    clusterChanged = True
    while (clusterChanged and iter>0):
        iter -=1
        clusterChanged = False
        for i in xrange(m):#for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            for j in xrange(k):
                distJI = distMeas(clustercents[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
#         print clustercents
#         print clusterAssment
        for cent in xrange(k):#recalculate clustercents
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
#             print 'class_ %d:  %d'%(cent, len(ptsInClust))
            if len(ptsInClust)>0:
#                 print mean(ptsInClust,axis=0)
                clustercents[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean
#     print 'iteration number: %d' % (100-iter)
    return clustercents
def biKmeans(dataSet, k,distMeans = distEclud):  
    numSamples = dataSet.shape[0]  
    # first column stores which cluster this sample belongs to,  
    # second column stores the error between this sample and its centroid  
    clusterAssment = mat(zeros((numSamples, 2)))  
  
    # step 1: the init cluster is the whole data set  
    centroid = mean(dataSet, axis = 0).tolist()[0]  
    centList = [centroid]  
    for i in xrange(numSamples):  
        clusterAssment[i, 1] = distMeans(mat(centroid), dataSet[i, :])**2  
  
    while len(centList) < k:  
        # min sum of square error  
        minSSE = inf  
        numCurrCluster = len(centList)  
        # for each cluster  
        for i in range(numCurrCluster):  
            # step 2: get samples in cluster i  
            pointsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]  
  
            # step 3: cluster it to 2 sub-clusters using k-means  
            centroids, splitClusterAssment = kMeans(pointsInCurrCluster, 2,distMeans)  
  
            # step 4: calculate the sum of square error after split this cluster  
            splitSSE = sum(splitClusterAssment[:, 1])  
            notSplitSSE = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])  
            currSplitSSE = splitSSE + notSplitSSE  
  
            # step 5: find the best split cluster which has the min sum of square error  
            if currSplitSSE < minSSE:  
                minSSE = currSplitSSE  
                bestCentroidToSplit = i  
                bestNewCentroids = centroids.copy()  
                bestClusterAssment = splitClusterAssment.copy()  
  
        # step 6: modify the cluster index for adding new cluster  
        bestClusterAssment[nonzero(bestClusterAssment[:, 0].A == 1)[0], 0] = numCurrCluster  
        bestClusterAssment[nonzero(bestClusterAssment[:, 0].A == 0)[0], 0] = bestCentroidToSplit  
  
        # step 7: update and append the centroids of the new 2 sub-cluster  
        centList[bestCentroidToSplit] = bestNewCentroids[0, :]  
        centList.append(bestNewCentroids[1, :])  
  
        # step 8: update the index and error of the samples whose cluster have been changed  
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentroidToSplit), :] = bestClusterAssment  
  
    print 'Congratulations, cluster using bi-kmeans complete!'  
    return mat(centList), clusterAssment
    
# if __name__ == '__main__':  
#     path = "E:/SinaSIFT"
#     files = get_file_list(path)
#     data0 = readData(files[0])
#     k=10
#     centroid = randcenters(data0,k)
#     for filename in files:
#         print filename
#         data = readData(filename)
#         centroid,clusterAssment = kMeans(data,centroid,k,iter=100)
#     print centroid
    #print clusterAssment
    
    
    
    
    
    