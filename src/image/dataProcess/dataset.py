# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年4月15日
time: 上午9:46:05
Function
'''
import cPickle as pickle
from src.image.dataProcess.data_pickle import *
from src.image.dataProcess.sift_cluster_center import *
from src.image.dataProcess.dataDispose import disposeSiftVisualWord

class dataset(object):
    '''
    userData 用户的数据
    LowLevelFeature = 低层特征  42维
    SiftData = 图像的sift特征  128维
    SiftVisualWord = sift聚类后的统计特征 50维
    LowVisualWord = 低层特征（只对用户有效） 图像低层特征聚类后对用户的统计特征 427维
    Target 用户标签，分别是：uid,gender,edu,terminal,act_level,act_habit,f_inf,P_inf
    '''
    userData=None
    '''
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
    userdata[17]    用户微博影响因子1             （评论数+1)*(转发数+1)*(点赞数+1)/(粉丝数+1))      
    userdata[18]    用户微博影响因子2  （评论数+0.5)*(转发数+0.5)*(点赞数+0.5)/(粉丝数+0.5))
    userdata[19]    用户微博影响因子3  （评论数+1)+(转发数+1)+(点赞数+1))/(粉丝数+1))
    userdata[20]    用户微博影响因子4  （评论数+转发数+点赞数) 
    userdata[21]    用户微博影响因子(权重法) （评论数*wight评论)+(转发数*weight转发)+(点赞数*weight点赞)+(粉丝数*weight粉丝))      
    userdata[22]    图片大小的比例
    '''
    LowLevelFeature = None #42维
    SiftData =None
    SiftVisualWord = None
    LowVisualWord =None
    Target=None
    def __init__(self):
        '''
        '''
    def loaddataset(self,path):
        return pickle.load(open(path,'rb'))
    def dumpdataset(self,path):
        pickle.dump(self, open(path,'wb'), True)
def mergedata():
    datapath = "E:/Desktop/Image/SVMData/sina/data/user_img_sift.data"
    lowfeaturePath = "E:/Desktop/Image/SVMData/sina/data/lowfeaturevisulword.data"
    siftdataPath = "E:/Desktop/Image/SVMData/sina/data/siftvisulword.data"
    datasetpath = "E:/Desktop/Image/SVMData/sina/data/dataset.data"
   
    
    
    alldata = dataset()  
    UserImgSift = dataPickle().loaddataset(datapath)
    alldata.LowLevelFeature = UserImgSift.data.T[:42].T
    alldata.SiftData = UserImgSift.siftdata.T[:384].T
    col_row = UserImgSift.siftdata.T[-1]
    alldata.userData = UserImgSift.data.T[42:].T
    alldata.Target = UserImgSift.target
    alldata.userData = np.column_stack((alldata.userData,col_row))
    siftWord = cluster().loaddataset(siftdataPath)
    siftworduid,siftvisual = disposeSiftVisualWord(siftWord)
    lowfeature = cluster().loaddataset(lowfeaturePath)
    lowvisualuid = lowfeature.centers
    lowvisualdata = lowfeature.visualword
    uids = alldata.Target['uid']
    siftvisualtemp =[]
    lowvisualtemp =[]
    for i in range(len(alldata.userData)):
        
        uid =uids[i]
        lowindex = lowvisualuid.index(uid)
        lowvisualtemp.append(lowvisualdata[lowindex].tolist())
        siftindex = siftworduid.index(uid)
        siftvisualtemp.append(siftvisual[siftindex].tolist())
#         print i,lowindex
        
    alldata.LowVisualWord = np.array(lowvisualtemp)
    alldata.SiftVisualWord = np.array(siftvisualtemp)
    alldata.dumpdataset(datasetpath) 
    
if __name__ == '__main__':
    
    mergedata()
    

    
    
    print '------------------end!--------------------'
    
    
        