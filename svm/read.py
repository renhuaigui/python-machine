# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2015年10月23日
time: 上午10:56:29
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
        lineArr = line.strip().split()
        label.append(float(lineArr[0]))
#         print(lineArr)
        for i in lineArr[1:]:
            t = i.split(':')
            li.append(float(t[-1]))            
        dataset.append(li)
        li = []

    return dataset,label  
# dataset ,lable = readData("D:/Desktop/Image/SVMData/gender_predict.txt")
