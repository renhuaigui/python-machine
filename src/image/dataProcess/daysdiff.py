# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年1月11日  
time: 下午12:20:07
Function: 计算日期时间戳，注册到的天数
'''
import datetime

def strtodatetime(datestr,format):       
    return datetime.datetime.strptime(datestr,format)

def daysdiff(beginDate,endDate): 
    format="%Y-%m-%d"; 
    bd=strtodatetime(beginDate,format) 
    ed=strtodatetime(endDate,format)     
    oneday=datetime.timedelta(days=1) 
    count=0
    while bd!=ed: 
        ed=ed-oneday 
        count+=1
    return count


if __name__ == '__main__':
    beginDate = '2008-1-1'
    endDate = '2009-1-1'
    print "%s到%s相差 %s天" % (beginDate,endDate,daysdiff(beginDate, endDate))
