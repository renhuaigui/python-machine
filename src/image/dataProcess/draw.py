# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年4月15日
time: 下午8:54:06
Function: 
'''
from src.image.dataProcess.dataset import *
from matplotlib.font_manager import FontProperties
from pylab import *
import numpy as np

def pltTimeDistribution(data):
    arr = data[argsort(data)] 
    dic={}
    di =[]
    for item in arr:
        if item in dic.keys():
            dic[item]+=1.0
        else:
            dic[item]=1.0
    dic2 = sorted(dic.iteritems(),key=lambda d:d[0])

    for i in range(len(dic2)):
        di.append([dic2[i][0],dic2[i][1]]) 
    di = np.array(di)
    p = di.T[1]/di.T[1].sum()
    p2 = di.T[1]/len(arr)
    cdfp = cumsum(p)
#     print mean(p)
#     print sum(p)
#     print std(p)
#     ent = 0
#     for i in range(len(p)):
#         ent -= math.log(p[i])*p[i]
#     print ent
    fig = plt.figure(figsize=(8,6))
   
    ylabel(r'$\rm{CDF }$', fontsize=16)
    xlabel(r'$\rm{Post\ Time}$', fontsize=16)
    plt.plot(di.T[0],cdfp,color='r')
#     savefig("./picture/DistributionTime.eps")
    plt.show()
    plt.close()
    
    fig = plt.figure(figsize=(8,6))
    ylabel(r'$\rm{Person Number}$', fontsize=16)
    xlabel(r'$\rm{Post\ Time}$', fontsize=16)
    plt.hist(di.T[0],48,color='b')
#     savefig("./picture/DistributionTimePerson.eps")
    plt.show()
    plt.close()


if __name__ == '__main__':
    datapath = "E:/Desktop/Image/SVMData/sina/data/dataset.data"
    dt = dataset()
    dataset = dt.loaddataset(datapath)
    userdata = dataset.userData
    
    data = userdata.T[3]
    pltTimeDistribution(data)
#     print gender
#     for i in gender:
#         print i