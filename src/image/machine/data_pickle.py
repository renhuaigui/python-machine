# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年3月7日
time: 上午10:00:16
Function: 
'''
import time
import os
from numpy import *
import cPickle as pickle
# import sys
# reload(sys)
# sys.setdefaultencoding( "utf-8" )

class dataPickle(object):
    data = None
    def __init__(self,data,target=None):
        self.data = data
        self.target = target
    '''
        其中dataset.target:
    target[0]    图片名称
    target[1]    图片的mid
    target[2]    图片的uid
    target[3]    性别：男(1) ,女(0)
    target[4]    活动时间：0-白天，1-晚上
    target[5]    客服端标签：1-手机，0-网页和其他方式
    target[6]    学历：0-大学，1-其他学历或空缺

        其中 dataset.data:
    data[0:41]  是图片的低层特征
    data[42]    是图片转发数
    data[43]    图片评论数
    data[44]    图片点赞数 
    data[45]    发图的时间 （对应活动时间）
    data[46]    发图客服端 （对应客服端标签）
    data[47]    图片微博包含的图片总量
    data[48]    发图用户的性别
    data[49]    发图用户的学历 （对应学历标签）
    data[50]    用户的注册时长
    data[51]    用户的等级
    data[52]    用户的勋章数
    data[53]    用户的关注数
    data[54]    用户的粉丝数
    data[55]    用户的微博数
    data[56]    一年发图总数
    data[57]    图片微博数 
    data[58]    微博频率（微博数/注册时长）   
    data[59]    用户微博影响因子
    data[60]
    data[61]
    data[62]
    data[63]
    '''
     
     
def readsiftdata(path):
    rf =open(path)
    data =[]
    for i in rf.readlines():
        #print i.strip()
        data.append(float(i))
#     print len(data)
#     print data
    return data 

def data_fomat(dataPath,userPath,contentPath):
    '''
    userfile like:uid,gender,age,regNum,career,edu,label,grade,bagdeNum,followNum,fanNum,postNum,postImgNum(oneYear)
        1259163900,0,1986年10月24日,1989,高管,大学,八卦 金融业 随遇而安,22,10,224,145,544,71
    '''
    data = []
    target = []
    userlist = {}
    contentlist = []
    
    df = open(dataPath) 
    for line in df.readlines(): 
        datatemp = line.strip().split(',') 
        target.append(datatemp[:2]) 
        data.append(map(float, datatemp[2:]))
    df.close()
    
    lf = open(userPath)
    for line in lf.readlines():
        labeltemp = line.strip().split(',')
        userlist[labeltemp[0]]= labeltemp[1:]
    lf.close()
    
    
    
    cf = open(contentPath)
    for line in cf.readlines():
        contenttemp = line.strip().split(',')
        contentlist.append(contenttemp)
    cf.close()
   
    flag =0
    for i in range(len(target)):#遍历图片数据的mid
        mid = target[i][1]
        print mid,i
        flag = 0
        for content in contentlist:
            if (mid == content[1]) :
                uid = content[0]
                flag = 1
                
                target[i].append(uid) #添加用户uid
                #print content
                for tk in content[2:]:
                    data[i].append(float(tk))#加入图片的数据
                #print len(data[i])
                #print content
                
                #print userlist[uid]         
                
                #添加性别标签
                target[i].append(int(userlist[uid][0])) 
                data[i].append(float(userlist[uid][0]))
                
                #添加活动时间标签
                postTime = float(content[5])
                if (postTime<19 and postTime>7):
                    target[i].append(0)
                else:
                    target[i].append(1)
                
                #添加客服端标签       
                target[i].append(int(content[6])) 
                
                #添加教育背景标签
                education = userlist[uid][4]
                if education == '大学':
                    #print education
                    target[i].append(0)
                    data[i].append(0.0)
                else:
                    #print education
                    target[i].append(1)
                    data[i].append(1.0)
                
                #添加用户注册时长数据
                data[i].append(float(userlist[uid][2]))
                
                #添加用户等级、勋章数、关注数、粉丝数、微博数、一年内发图总数，图片微博总数
                for x in userlist[uid][6:]:
                    data[i].append(float(x))
                    
                #添加微博频率
                data[i].append(float(userlist[uid][10])/float(userlist[uid][2]))
                #添加用户影响因子（图片评论数+0.5）*（转发数+0.5）*（点赞数+0.5）/用户粉丝数
                data[i].append((data[i][42]+0.5)*(data[i][43]+0.5)*(data[i][44]+0.5)/(data[i][54]+0.5))
                
                #print data[i]
                #print target[i]
                break
        
        if(flag == 0):
            print i
            print(target[i][0]+"没有对应的标签")
            delete(data, i, 0)#删除对应的数据    
            delete(target, i,0)
    
    data = array(data)
    target = array(target)
    print data.shape
    print target.shape
    dataset=dataPickle(data,target)
    return dataset
  
          
if __name__ == '__main__':
    
    dataPath = "E:/Desktop/Image/SVMData/sina/data/image.txt"
    userPath = "E:/Desktop/Image/SVMData/sina/label/user_new.txt"
    contentPath= "E:/Desktop/Image/SVMData/sina/label/content.txt"
    writePath = "E:/Desktop/Image/SVMData/sina/data/img_pickle.data"
    userlabelpath ="E:/Desktop/Image/SVMData/sina/data/user_img_pickle.data"
    #dataset = data_fomat(dataPath, userPath, contentPath)
    #pickle.dump(dataset,open(writePath,"wb"),True)
   
    print '\n----------------------'
    print '---------End----------'
    print '----------------------'
    
    
    

    
        
        
        
        
        
    