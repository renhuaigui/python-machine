# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年3月8日
time: 下午6:54:29
Function: 
'''
import numpy as np
from data_pickle import *
import os
from calCDF import *

def dispose(data,target):
    '''
    Parameters
    ----------
    data：数据集
    target；数据集的标签
         计算每个用户的所有特征列的平均值
       返回用户对应的数据和用户ID
    '''
    userdata = {}
    label ={}
    dataset = []
    uidlist = target.T[2]
#     pltCDF(data.T[45][:])
    for i in range(len(uidlist)):
        #print uidlist[i]
        if uidlist[i] in userdata:
            userdata[uidlist[i]].append(data[i].tolist())
        else :
            userdata[uidlist[i]] = [data[i].tolist()]
    user = []
    for key in userdata:
        #print key
        userdata[key] = np.array(userdata[key])
        
        userdata[key] = map(np.mean, zip(userdata[key].T))
#         print userdata[key]
        dataset.append(userdata[key])
        
        user.append(int(key))
        #print userdata[key]
    user = map(int,user)
    label['uid'] = user
    dataset = np.array(dataset)
    
    gender = map(int,dataset.T[48][:])
    label['gender']= gender#性别标签
    #print gender_label
    edu = map(int,dataset.T[49][:])
    label['edu']=edu#学历标签
    #print edu
    term = [int(tk+0.5) for tk in dataset.T[46]]
    label['term']=term#终端标签

    #print term
    act_time = dataset.T[45][:]
   
    act_time_two = []#活动时间分两类
    act_time_three = []#活动时间分三类
    for i in range(len(act_time)):
        if (act_time[i]>13.5 and act_time[i]<=17.5):
            act_time_two.append(0) 
        else :
            act_time_two.append(1)
        if(act_time[i]>0. and act_time[i]<=13.0):
            act_time_three.append(0)
        elif(act_time[i]>13.0 and act_time[i]<=16.0):
            act_time_three.append(1)
        else:
            act_time_three.append(2)
    label['act_habit_2']= act_time_two
    label['act_habit_3']= act_time_three #活动习惯（活动时间）标签
    
    all_img_num = dataset.T[56]
    act_level=[]
    act_level_two =[]
    for i in range(len(all_img_num)):
        if all_img_num[i]<50:
            act_level_two.append(0)
        else:
            act_level_two.append(1)
        if all_img_num[i]<=10:
            act_level.append(0)
        elif (all_img_num[i]>10 and all_img_num[i]<=100):
            act_level.append(1)
        else :
            act_level.append(2)
    label['act_level'] = act_level #添加活跃程度
    label['act_level_two'] = act_level_two
    
#     pltCDF(dataset.T[45][:])
    influence = []
    influence_two =[]
    user_influence =[]
    f_influence =[]
 
    from sklearn import preprocessing
    min_max_scaler =preprocessing.MinMaxScaler()   
    forwardProb =min_max_scaler.fit_transform(dataset.T[42][:])
    commentProb = min_max_scaler.fit_transform(dataset.T[43][:])
    heartProb = min_max_scaler.fit_transform(dataset.T[44][:])
    picturNumProb = min_max_scaler.fit_transform(dataset.T[54][:])
    followProb = min_max_scaler.fit_transform(dataset.T[56][:])
    

    forwardEntropy = calEntropy(dataset.T[42][:])
    commentEntropy = calEntropy(dataset.T[43][:])
    heartEntropy = calEntropy(dataset.T[44][:])
    picturNumEntropy =calEntropy(dataset.T[54][:])
    followEntropy =calEntropy(dataset.T[56][:])

    forwardWight = (1-forwardEntropy)/(5-forwardEntropy-commentEntropy-heartEntropy-picturNumEntropy-followEntropy)
    commentWight = (1-commentEntropy)/(5-forwardEntropy-commentEntropy-heartEntropy-picturNumEntropy-followEntropy)
    heartWight = (1-heartEntropy)/(5-forwardEntropy-commentEntropy-heartEntropy-picturNumEntropy-followEntropy)
    picturNumWight = (1-picturNumEntropy)/(5-forwardEntropy-commentEntropy-heartEntropy-picturNumEntropy-followEntropy)
    followWight = (1-followEntropy)/(5-forwardEntropy-commentEntropy-heartEntropy-picturNumEntropy-followEntropy)
    print forwardWight,commentWight,heartWight,picturNumWight,followWight
    influencetr = forwardWight*forwardProb+commentProb*commentWight+heartProb*heartWight+picturNumProb*picturNumWight+followProb*followWight
    print len(influencetr)
    
#     for i in range(len(dataset.T[54])):
#         print user[i],dataset.T[54][i],dataset.T[59][i]
    print dataset.T[54]
    pltCDF(influencetr)
    for i in range(len(user)):
        if dataset.T[59][i]<=0.01:
            influence_two.append(0)
        else:
            influence_two.append(1)
        if dataset.T[59][i]<=0.001 :
            influence.append(0)
        elif dataset.T[59][i]>0.001 and dataset.T[59][i]<=0.1:
            influence.append(1)
        else:
            influence.append(2)
        if influencetr[i]<=0.03:
            user_influence.append(0)
        else:
            user_influence.append(1)
        if dataset.T[54][i]<=10000:
            f_influence.append(0)
        else:
            f_influence.append(1)        
    label['influence'] = influence    
    label['influence_two'] = influence_two
    label['user_influence'] = user_influence
    label['f_influence'] = f_influence
    print influencetr
    print len(dataset),len(dataset[0]),len(influencetr)
#     pltCDF(influencetr)
    dataset =np.column_stack((dataset,influencetr))
    print len(dataset),len(dataset.T[0])
    print dataset.T[63]
    '''
    dataset:返回处理后的数据
    label：返回处理后的标签
    '''   
    return dataset,label

def calEntropy(data):
    dataProb = data/data.sum()
    print data.sum()
    entopy = 0
    for i in dataProb:
#         print i
        if i !=0:            
            entopy -= i*math.log(i)
    return entopy

def readData(path):
    '''
        读取文件path的内容
        返回数据矩阵
    '''
    data= []
    rf = open(path)
    for line in rf.readlines():
        tokens = line.strip().split(',')
        data.append(map(float,tokens))
    return mat(data)

def get_file_list(path):
    #获取文件夹下所有文件（包括子文件）
    current_files = os.listdir(path)
    print
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

if __name__ == '__main__':
    
#     siftpath = 'E:/SinaSIFT'
#     print get_files(siftpath)
#     print len(get_files(siftpath)) 
#     print get_file_list('E:/SinaSIFT/1003763994/')


    #统计每个用户的数据均值
    path = "E:/Desktop/Image/SVMData/sina/data/img_pickle.data"
    writePath = "E:/Desktop/Image/SVMData/sina/data/user_img_pickle1.data"
    dataset = pickle.load(open(path,"rb"))
    data,target=dispose(dataset.data, dataset.target)
    dataset=dataPickle(data,target)
#     pickle.dump(dataset,open(writePath,"wb"),True)

    
    print '\n----------------------'
    print '---------End----------'
    print '----------------------'