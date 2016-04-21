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
import dataDispose
# import sys
# reload(sys)
# sys.setdefaultencoding( "utf-8" )

class dataPickle(object):
    data = None
    target = None
    siftdate = None
    userdata =None
    def __init__(self):
        ""
    def pickledata(self,data=None,target=None,siftdata = None,userdata=None):
        self.data = data
        self.target = target
        self.siftdata = siftdata
        self.userdata = userdata
    def loaddataset(self,path):
        return pickle.load(open(path,'rb'))
    def dumpdataset(self,path):
        pickle.dump(self, open(path,'wb'), True)
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
    siftdata[384]  是图片的大小比例 
    
        其中 dataset.userdata:
    userdata[0]    是图片转发数
    userdata[1]    图片评论数
    userdata[2]    图片点赞数 
    userdata[3]    发图的时间 （对应活动时间）
    userdata[4]    发图客服端 （对应客服端标签）
    userdata[5]    图片微博包含的图片总量
    userdata[6]    发图用户的性别
    userdata[7]    发图用户的学历 （对应学历标签）
    userdata[8]    用户的注册时长
    userdata[9]    用户的等级
    userdata[10]    用户的勋章数
    userdata[11]    用户的关注数
    userdata[12]    用户的粉丝数
    userdata[13]    用户的微博数
    userdata[14]    一年发图总数
    userdata[15]    图片微博数 
    userdata[16]    微博频率（微博数/注册时长）   
    userdata[17]    用户微博影响因子1
    userdata[18]    用户微博影响因子1
    userdata[19]    用户微博影响因子1
    userdata[20]    用户微博影响因子1    
    userdata[21]    用户微博影响因子(权重法)
    userdata[22]    图片大小的比例

        
        
        其中 dataset.data:
    data[0:41]  是图片的低层特征
    data[42]    图片转发数
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
    siftdata = []
    df = open(dataPath) 
    for line in df.readlines(): 
        datatemp = line.strip().split(',') 
        target.append(datatemp[:2]) 
#         print datatemp[0]
#         print len(datatemp[2:44])
#         print len(datatemp[44:])
        data.append(map(float, datatemp[2:44]))
        siftdata.append(map(float, datatemp[44:]))
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
                data[i].append((data[i][42]+1)*(data[i][43]+1)*(data[i][44]+1)/(data[i][54]+1))
                data[i].append((data[i][42]+0.5)*(data[i][43]+0.5)*(data[i][44]+0.5)/(data[i][54]+0.5))
                data[i].append(((data[i][42]+1)+(data[i][43]+1)+(data[i][44]+1))/(data[i][54]+1))
                data[i].append(data[i][42]+data[i][43]+data[i][44]) 
                if len(data[i]) !=63:
                    print '出错了'
                    time.sleep(10)
                #print data[i]
                #print target[i]
                break
        
        if(flag == 0):
            print i
            print(target[i][0]+"没有对应的标签")
            delete(data, i, 0)#删除对应的数据    
            delete(target, i,0)
            delete(siftdata,i,0)
    
    data = array(data)
    target = array(target)
    siftdata = array(siftdata)
    userdata = data.T[42:60].T
    print data.shape
    print target.shape
    print siftdata.shape
    print userdata.shape
    dataset=dataPickle.pickledata(data,target=target,siftdata=siftdata,userdata=userdata)
    return dataset

def appendSIFT(dataset,siftdataPath):
    data = dataset.data
    target = dataset.target
    mid = target.T[1]
    siftmid = []
    siftdata = []
    rf = open(siftdataPath)

    #print mid
    for line in rf.readlines():
        datatemp = line.strip().split(',') 
        siftmid.append(datatemp[1]) 
        siftdata.append(map(float, datatemp[2:]))
    print len(siftmid),len(siftdata)      
    
if __name__ == '__main__':
    
    dataPath = "E:/Desktop/Image/SVMData/sina/data/image.txt"
    userPath = "E:/Desktop/Image/SVMData/sina/label/user_new.txt"
    contentPath= "E:/Desktop/Image/SVMData/sina/label/content.txt"
    writePath = "E:/Desktop/Image/SVMData/sina/data/img_pickle.data" 
    writeSiftPath = "E:/Desktop/Image/SVMData/sina/data/img_pickle.Siftdata"
    userlabelpath ="E:/Desktop/Image/SVMData/sina/data/user_img_pickle.data"
    siftdataPath = "E:/Desktop/Image/SVMData/sina/data/siftimage.txt"
#     dataset = data_fomat(dataPath, userPath, contentPath)
#     pickle.dump(dataset,open(writeSiftPath,"wb"),True)
    dataset = pickle.load(open(writeSiftPath,'rb'))
   
    dat = dataPickle()
    dat.data = dataset.data
    dat.userdata = dataset.userdata     
    dat.siftdata = dataset.siftdata
    dat.target = dataset.target
    dat.dumpdataset(writeSiftPath)
    data = dat.loaddataset(writeSiftPath)
    print data.data.shape
    


    print '\n----------------------'
    print '---------End----------'
    print '----------------------'
    
    
    

    
        
        
        
        
        
    