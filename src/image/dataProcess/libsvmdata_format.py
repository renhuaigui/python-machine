# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2015年10月26日
time: 下午4:22:44
Function:从文件总读取数据加上标签保存成libsvm数据格式的文件可以
                     可以是多个标签
'''
from sklearn.datasets.svmlight_format import dump_svmlight_file
import numpy as np
from scipy import double
import codecs
import xlrd
import re
import time 


def wechat_fomat(dataPath,labelPath,writeGenderPath,writeLocPath):
    '''微信数据格式化'''
    imagename = []#每行数据所对应的图片么
    data = []# 数据矩阵
    genderlabel = []#性别标签
    loclabel = []#位置标签
    labelfile = []
    
    #注意读取的格式编码！！！，有中文时字符编码是uft-8的菜可以识别，
    #可以在eclipse建立普通文件复制内容过来就可以解决
    ''''读取数据'''''
    f = codecs.open(dataPath)  
    for line in f.readlines(): 
        tokens = line.strip().split(' ')  
        imagename.append(tokens[0])
        data.append([double(tk) for tk in tokens[1:]])
    f.close()
    imagename = np.array(imagename)
    data = np.array(data)
    print imagename 
    '''''读取标签'''
    labelf = codecs.open(labelPath)  
    for line in labelf.readlines(): 
        tokens = line.strip().split(' ')  
        labelfile.append([tk for tk in tokens[:]])
    # print labelfile
    
    flag = 0
    '''填入标签'''''
    for i in range(0,len(imagename)):
        name = imagename[i]
        flag = 0
        for li in labelfile:
    #         print li[3]
            if(name == li[3]):
                flag = 1
                if(li[1] == '女'):
                    genderlabel.append(0)
                else:
                    genderlabel.append(1)
                if(li[5] == '2'):
                    loclabel.append(0)
                else:
                    loclabel.append(1)
        if(flag == 0):
            print i
            print(name+"没有对应的标签")
            np.delete(data, i, 0)#删除对应的数据
    # print loclabel  
    # label = np.array(label)
    labelf.close()
    
    ''''稀疏矩阵化数据'''
    data = np.array(data)
    genderlabel = np.array(genderlabel)
    loclabel = np.array(loclabel)
    
    '''查看数据是否一致大小
    如果结果不一致说明标签和数据不匹配.
    '''
    print data.shape[0]
    print genderlabel.shape[0]
    print loclabel.shape[0] 
    ''''将libsvm格式数据写到文件'''
    dump_svmlight_file(data, genderlabel,writeGenderPath,zero_based=False)
    dump_svmlight_file(data, loclabel,writeLocPath,zero_based=False)
    print ("Wechat format End!")
    
    
    
def feature_format(sinadataPath,userPath,contentPath,sinaGenderPath):
    '''性别标签格式化'''
    imagename = []#每行数据所对应的图片么
    data = []# 数据矩阵
    contentlist = []#微博列表
    genderlabel = []#性别标签
    userlist = [] #用户列表
    
    #注意读取的格式编码！！！，有中文时字符编码是uft-8的菜可以识别，
    #可以在eclipse建立普通文件复制内容过来就可以解决
    ''''读取数据'''''
    f = codecs.open(sinadataPath)  
    for line in f.readlines(): 
        datatemp = line.strip().split(',')  
        imagename.append(datatemp[1])
        data.append([double(tk) for tk in datatemp[2:]])
    f.close()
    imagename = np.array(imagename)
    data = np.array(data)
    #print imagename 
    #print data
    print '''读取发图微博'''
    contentf = codecs.open(contentPath)  
    for line in contentf.readlines(): 
        contenttemp = line.strip().split(',')  
        #print contenttemp[1]
        contentlist.append([tk for tk in contenttemp[:]])
    contentf.close()
    print '''读取用户列表'''
    userf = codecs.open(userPath)  
    for line in userf.readlines(): 
        usertemp = line.strip().split(',')  
        #print usertemp[1]
        userlist.append([tk for tk in usertemp[:]])
    userf.close()
    
    print '''填入标签'''
    for i in range(0,len(imagename)):
        name = imagename[i]
        print name
        flag = 0
        for li in contentlist:
            #print li
            if(name == li[1]):
                for user in userlist:
                    #print user
                    if(user[0] == li [0]):
                        flag = 1
                        #print user[1]
                        if(user[1] == '女'):
                            genderlabel.append(0)
                        else:
                            genderlabel.append(1)
                        break
                break
            #print genderlabel
        if(flag == 0):
            print i
            print(name+"没有对应的标签")
            np.delete(data, i, 0)#删除对应的数据
            
    genderlabel = np.array(genderlabel)  

    
    print genderlabel
    print data.shape[0] 
    print genderlabel.shape[0]
    print ''''构建libsvm数据'''
    dump_svmlight_file(data, genderlabel,sinaGenderPath,zero_based=False)
     
def user_format(userPath,userSavePath):
    '''给粉丝|关注|微博数加性别标签'''
    print '''读取用户列表'''
    userlist =[]
    gender =[]
    t=[]
    userf = codecs.open(userPath)  
    for line in userf.readlines(): 
        usertemp = line.strip().split(',')  
        userlist.append([tk for tk in usertemp[:]])
    userlist = np.array(userlist)
    data = userlist.T[4:].T    
    genderlabel = userlist.T[1].T
    for i in range(0,len(genderlabel)):
        if genderlabel[i] =='女':
            gender.append(0)
        else:
            gender.append(1)
        t.append([double(tk) for tk in data[i][:]])
 
    data = np.array(t)
    genderlabel = np.array(gender)   
    print len(data[0])
    print len(genderlabel)
    dump_svmlight_file(data, genderlabel,userSavePath,zero_based=False)
    userf.close()
    

def sina_content_format(contentPath,userPath,contentSavePath):
    '''图片微博加标签'''
    data = []
    userlist =[]
    contentlist =[]
    genderlabel= []
    contentf = codecs.open(contentPath) 
    print '''读取发图微博'''
    for line in contentf.readlines(): 
        contenttemp = line.strip().split(',')  
        #print contenttemp[1]
        contentlist.append([tk for tk in contenttemp[:]])
    contentf.close()
   
    print '''读取用户列表'''
    userf = codecs.open(userPath)  
    for line in userf.readlines(): 
        usertemp = line.strip().split(',')  
        #print usertemp[1]
        userlist.append([tk for tk in usertemp[:]])
    userf.close()
    
    print '''填入数据和标签'''
    for i in range(len(contentlist)):
        temp = []
        temp=[int(tk) for tk in contentlist[i][2:5]]
        s = contentlist[i][5].split(" ")
        t = s[1].split(":")
        temp.append(int(t[0]))
        temp.append(int(contentlist[i][7]))
        #填入数据
        data.append(temp)
        #填入标签
        for user in userlist:
            if(user[0]==contentlist[i][0]):
                if(user[1] == '女'):
                    genderlabel.append(0)
                else:
                    genderlabel.append(1)
                break
    
    data = np.array(data)
    genderlabel = np.array(genderlabel)  
    
    print len(data)  
    print len(genderlabel)
    dump_svmlight_file(data, genderlabel,contentSavePath,zero_based=False)

'''发图客服端标签：1手机类，2网页类，3应用游戏等类，0其他'''
    
def term_label_format(imagePath,termPath,termWritePath):
    ''''加终端标签'''
    data = []
    termlabel = []
    termlist = []
    imagename =[]
    contentlist= []
    img = codecs.open(imagePath)  
    
    for line in img.readlines(): 
        datatemp = line.strip().split(',')  
        imagename.append(datatemp[1])
        data.append([double(tk) for tk in datatemp[2:]])
    img.close()
    imagename = np.array(imagename)
    data = np.array(data)
    
    comf = codecs.open(termPath)
    for line in comf.readlines():
        termlisttemp = line.strip().split(",")
        termlist.append(termlisttemp)
    comf.close()
    
    contentf = codecs.open(contentPath)  
    for line in contentf.readlines(): 
        contenttemp = line.strip().split(',')  
        #print contenttemp[1]
        contentlist.append([tk for tk in contenttemp[:]])
    contentf.close()
    
    print '''填入标签'''
    for i in range(0,len(imagename)):
        name = imagename[i]
        flag = 0
        for li in contentlist:
            #print li
            if(name == li[1]):
                for term in termlist:
                    if(term[0] == li [6]):
                        flag = 1
                        termlabel.append(int(term[1]))
                        print i,name,term[0],term[1]
                        break
                break
        if(flag == 0):
            print i
            print(name+"没有对应的标签")
            del data[i]#删除对应的数据
    termlabel = np.array(termlabel)
    print termlabel
    print len(data)
    print len(termlabel)
    dump_svmlight_file(data, termlabel,termWritePath,zero_based=False)
     
def cutUselessFeatures(Path,delline,writePath):
    """
        把区分度不大的特征删除
    """
    from sklearn.datasets import load_svmlight_file
    data, target = load_svmlight_file(Path)
    deldata=data.toarray()
    for i in delline:
        print i
        deldata = np.delete(deldata, i, 1)#删除第i列的元素值
    
    
    dump_svmlight_file(deldata, target,writePath,zero_based=False)

def gender_label_format(userPath,contentPath,DataPath,WritePath):
    '''加性别标签'''
    data = []
    label = []
    userlist =[]
    imagename =[]
    contentlist= []
    print'''导入数据'''
    img = codecs.open(DataPath)  
    for line in img.readlines(): 
        datatemp = line.strip().split(',')  
        imagename.append(datatemp[1])
        data.append([double(tk) for tk in datatemp[2:]])
    img.close()
    imagename = np.array(imagename)
    data = np.array(data)
    print'''导入用户信息'''
    userf = codecs.open(userPath)  
    for line in userf.readlines(): 
        usertemp = line.strip().split(',')  
        #print usertemp[1]
        userlist.append([tk for tk in usertemp[:]])
    userf.close()
    print'''导入图片信息'''
    contentf = codecs.open(contentPath)  
    for line in contentf.readlines(): 
        contenttemp = line.strip().split(',')  
        #print contenttemp[1]
        contentlist.append([tk for tk in contenttemp[:]])
    contentf.close()
    
    print '''填入标签'''
    for i in range(0,len(imagename)):
        name = imagename[i]
        print name
        flag = 0
        for li in contentlist:
            #print li
            if(name == li[1]):
                for user in userlist:
                    #print user
                    if(user[0] == li [0]):
                        flag = 1
                        print user[1]
                        if(user[1] == '女'):
                            label.append(0)
                        else:
                            label.append(1)
                        break
                break
            #print genderlabel
        if(flag == 0):
            print i
            print(name+"没有对应的标签")
            np.delete(data, i, 0)#删除对应的数据            
    label = np.array(label)
    print label
    print len(data)
    print len(label)
    dump_svmlight_file(data, label,WritePath,zero_based=False)
    
    
if __name__ == '__main__':  
    '''文件路径'''
    dataPath = "E:/Desktop/Image/Original/wang_wechat_data.txt"
    labelPath = "./wang_infor"
    writeGenderPath = "E:/Desktop/Image/SVMData/wang_gender_libsvmdata.txt"
    writeLocPath = "E:/Desktop/Image/SVMData/wang_Loc_libsvmdata.txt"
    
    imagePath = "E:/Desktop/Image/SVMData/sina/data/image.txt"
    userPath = "E:/Desktop/Image/SVMData/sina/label/user.txt"
    contentPath= "E:/Desktop/Image/SVMData/sina/label/content.txt"
    
    sinaGenderPath = "E:/Desktop/Image/SVMData/sina/data/sina_gender_libsvmdata.txt"
    
    termPath = "E:/Desktop/Image/SVMData/sina/label/comfrom.txt"
    termWritePath = "E:/Desktop/Image/SVMData/sina/data/comfrom_libsvmdata.txt"
    
    siftPath = "E:/Desktop/Image/SVMData/sina/data/siftimage.txt"
    siftWritePath = "E:/Desktop/Image/SVMData/sina/data/sift_gender_libsvmdata.txt"
    
    
    
    '''图片相关信息数据libsvm化'''
    #sina_content_format(contentPath,userPath,"E:/Desktop/Image/SVMData/sina/data/contentGenderLibsvmdata.txt")
    
    '''用户数据libsvm化'''
    #user_format(userPath,'E:/Desktop/Image/SVMData/sina/userGenderLibsvmdata.txt')
    
    '''图片特征数据libsvm化'''
    #feature_format(imagePath,userPath,contentPath,sinaGenderPath)
    
    '''发图终端libsvm化'''
#     start_time = time.time()
#     term_label_format(imagePath, termPath, termWritePath)
#     print '运行时间： %fs!' % (time.time() - start_time)

    '''sift特征libsvm化'''
    gender_label_format(userPath, contentPath, siftPath, siftWritePath)

    
    
#     delPath = "E:/Desktop/Image/SVMData/sina/sina_gender_libsvmdata_scale.txt"
#     deline = [38,37,36,35,31,30,27,26,23,22,19,18,16,14,12,11,9,0]
#     delwritePath ="E:/Desktop/Image/SVMData/sina/sina_gender_libsvmdata_scale_del.txt"
#     print 'Useless feature:',deline
#     cutUselessFeatures(delPath, deline, delwritePath)
    
    
    
    
    print "end!"






    