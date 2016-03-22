# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on ： 2015年9月11日
time ：下午2:17:47
Function: 
'''

def readData(Base):
    """
    read data return UID,X,y,
    
    """
    dataset = []
    label = []
    li = []
#     uidL = []
    fileIn = open(Base)  
    for line in fileIn.readlines(): 
        print line
#         lineArr = line.strip().split()
#         label.append(float(lineArr[0]))
#         print(lineArr)
#         for i in lineArr[1:]:
#             t = i.split(':')
#             li.append(float(t[-1]))            
#         dataset.append(li)
#         li = []
#         
        
        
#     print(dataset)     
#     with open(Base,'r') as fr:
#         for eline in fr:
#             li = eline.strip().split('\t')
#             print(li)
#            # li = [float(e) for e in li[:-1] + [li[-1].replace('\n','')]]
           # if int(li[yid])>0:
              #  uidL.append(int(li[0]))
#                 dataset.append([e for e in li[1:-1]])
#                 lable.append(int(li[0]))
    return dataset,label  
dataset ,lable = readData("E:/SinaSIFT/filelist.txt")
