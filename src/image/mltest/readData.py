# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2015年10月23日
time: 下午2:02:17
Function: 
'''

from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn import datasets
import MySQLdb
from _mysql import result

def readData(path):
    data, target = load_svmlight_file(path)
#     i=0
#     for value in target:
#         if value == 2:
#             target[i] = 0
#         i = i+1
    return data,target
def readDbData():
    data = []
    genderlable =[]
    
    try:
        conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='',db='sina_weibo',port=3306,charset="utf8")
        dataSQL = 'select * from image'                  
       
        cur=conn.cursor()
        gender = conn.cursor()
        dataSql=cur.execute(dataSQL)
        for li in cur.fetchmany(30):
            
            mid = li[2]
            print mid
            lableSQL = "select * from user where accountID =any\
                        (select uid from content where mid =any\
                        (select mid from image where mid ="+str(mid)+"))" 
            print lableSQL
            gender.execute(lableSQL)
           
            print gender.fetchone()
        gender.close()
        cur.close()
        conn.close()
    except MySQLdb.Error,e:
         print "Mysql Error %d: %s" % (e.args[0], e.args[1])    



