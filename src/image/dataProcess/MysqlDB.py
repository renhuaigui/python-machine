# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年1月11日
time: 下午1:24:21
Function: 
'''
import sys  
import MySQLdb
reload(sys)  
sys.setdefaultencoding('utf8')  

    
try:
    conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='',db='sina_weibo',port=3306)
    cur=conn.cursor()
    sql = 'select * from user'
    cur.execute(sql)
    cur.close()
    conn.close()
except MySQLdb.Error,e:
    print "Mysql Error %d: %s" % (e.args[0], e.args[1])
     
     
     