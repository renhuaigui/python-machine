# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年2月26日
time: 下午3:41:05
Function: 
'''
import codecs
import sys
import time
from src.image.dataProcess.daysdiff import daysdiff

def Zodiac(month, day):
    n = ('摩羯座','水瓶座','双鱼座','白羊座','金牛座','双子座','巨蟹座','狮子座','处女座','天秤座','天蝎座','射手座')
    d = ((1,20),(2,19),(3,21),(4,21),(5,21),(6,22),(7,23),(8,23),(9,23),(10,23),(11,23),(12,23))
    return n[len(filter(lambda y:y<=(month,day), d))%12]


def cal_Zodiac(userdata,readPath,writePath):
    endDate = '2015-10-01'
    '''''读用户数据''' 
    userlist = []
    
    userf = codecs.open(readPath)      
    for line in userf.readlines(): 
        usertemp = line.strip().split(',')  
        #print usertemp
        userlist.append([tk for tk in usertemp[:]])
    userf.close()
   
    writef=file(writePath,"a+")
    i=0
    for user in userlist:
        print user[0]
        age=''
        zodiac_sign=''
        time = user[2].split('-')
        '''计算年龄和星座'''
        if len(time)==1:
            if time[0]!='':
                zodiac_sign = time[0]
        if len(time)==2:
            zodiac_sign = Zodiac(int(time[0]), int(time[1]))
        if len(time)==3:
            age=2015-int(time[0])
            zodiac_sign = Zodiac(int(time[1]), int(time[0]))

        userlist[i].append(age)
        userlist[i].append(zodiac_sign)

        '''写入文件'''
        for temp in userlist[i]:
            writef.write(str(temp)+',')
        for data in userdata:
            if data[0] == userlist[i][0]:
                writef.write(data[1]+','+data[2])
                break       
        writef.write('\n')
        i=i+1
        
    writef.close()
    
def data_extraction(path):  
    '''
        提取等级，标签，勋章数
    '''
    userdata =[]
    userlist =[]
    userf = codecs.open(path)    
    for line in userf.readlines(): 
        usertemp = line.strip().split(';')  
        #print usertemp
        userlist.append([tk for tk in usertemp[:]])
    userf.close()
    num =0
    for user in userlist:
        bagde = []
        bagde = user[3].split(',')
        num= len(bagde)
        if user[3] == '':
            num = 0#如果为空值为0
        userdata.append([user[0],user[2],str(num)])
    return userdata

def label_statistic(readPath,writePath):
    dict = {}
    userlist = []
    f = codecs.open(readPath)
    for line in f.readlines():
        usertemp = line.strip().split(';')
        userlist.append([tk for tk in usertemp[:]])
    f.close()
    for user in userlist:
        label = []
        label = user[1].split(' ')
        for labels in label:
            if labels not in dict:
                dict[labels] = 1
            else:
                dict[labels] = dict[labels]+1;   
    dic = sorted(dict.iteritems(),key = lambda asd:asd[1],reverse = True)
   
    wf = codecs.open(writePath,"a+")
    for i in range(len(dic)):
        print '%s  %s' % (dic[i][0],dic[i][1])
        wf.write(str(dic[i][0]+' '+str(dic[i][1])+'\n'))
    wf.close()

def contentP(contentPath,termPath,writePath):
    contentlist=[]
    termlist={}
    tf = codecs.open(termPath)
    for line in tf.readlines():
        termtemp = line.strip().split(',')
        termlist[termtemp[1]] = termtemp[2]
    tf.close()
    rf = codecs.open(contentPath)
    for line in rf.readlines():
        contenttemp = line.strip().split(',')
        time = contenttemp[5].split(' ')[1].split(":")
        contenttemp[5] = float(time[0])+float(time[1])/60.0
        contenttemp[6] = termlist[contenttemp[6]]
        contentlist.append(contenttemp)
    wf = codecs.open(writePath,'w')
    for li in contentlist:
        for lin in li:
            wf.write(str(lin)+',')
        wf.write('\n')
    wf.close()
  
def userP(userPath,contentPath,writePath):
    userlist = {}
    contentlist = []
    uf = open(userPath)
    for line in uf.readlines():
        utemp = line.strip().split(';')
        bagdenum = len(utemp[8].split(','))
        if utemp[8] == '':
            bagdenum = 0
        utemp[8] = bagdenum
        utemp[3] = daysdiff(utemp[3],'2015-10-01')
        #print utemp
        #print utemp[8]
        utemp.append(0)
        utemp.append(0)
        #print len(utemp)
        #print utemp
        userlist[utemp[0]] = utemp[:]
        #time.sleep(0.5)
    uf.close()
    mid = 0
    cf = open(contentPath)
    for line in cf.readlines():
        contenttemp = line.strip().split(',')
        #print contenttemp
        userlist[contenttemp[0]][-2] = userlist[contenttemp[0]][-2]+int(contenttemp[7])#统计图片总数
        if mid != contenttemp[1]:
            userlist[contenttemp[0]][-1] = userlist[contenttemp[0]][-1]+1#统计图片微博总数
        else :
            print '重复l'
        mid = contenttemp[1]
        #contentlist.append(contenttemp)
    cf.close()   
    
    wf = open(writePath,'w')
    for key in userlist:
        for li in userlist[key]:
            wf.write(str(li)+',')
        wf.write('\n')
    wf.close()    
   
if __name__ == '__main__':
    userPath = "E:/Desktop/Image/SVMData/sina/label/user3.txt"
    userPath1= "E:/Desktop/Image/SVMData/sina/label/user_zodiac.txt"
    path = "E:/Desktop/Image/SVMData/sina/label/user1.txt"
    
    contentPath = "E:/Desktop/Image/SVMData/sina/label/content_old.txt"
    termPath ="E:/Desktop/Image/SVMData/sina/label/terminal.txt"
    contentWrite = "E:/Desktop/Image/SVMData/sina/label/content.txt"
#     userdata = data_extraction(path)
#     cal_Zodiac(userdata,userPath,userPath1)
    user_label_Path ='E:/Desktop/Image/SVMData/sina/label/user_label.txt'
    #label_statistic(path, user_label_Path)
    #contentP(contentPath, termPath, contentWrite)
    
    userupdate = "E:/Desktop/Image/SVMData/sina/label/user_new.txt"
    #userP(userPath, contentWrite, userupdate)
    
    print "**********End!**********"
    
    
    
    
    
    
