# -*- coding:utf-8 -*-
'''
Created on 2016年3月1日

@author: Administrator
'''
import os
import codecs
from numpy import *

def get_file_list(path):
    current_files = os.listdir(path)
    all_files = []
    for file_name in current_files:
        full_file_name = os.path.join(path, file_name)
        all_files.append(full_file_name)
 
        if os.path.isdir(full_file_name):
            next_level_files = get_file_list(full_file_name)
            all_files.extend(next_level_files)
 
    return all_files


def readData(path):
    data= []
    rf = open(path)
    for line in rf.readlines():
        tokens = line.strip().split(',')
        data.append(map(float,tokens))
    return mat(data)


if __name__ == '__main__':
    path = "E:/SinaSIFT"
    files = get_file_list(path)
    print files
    print len(files)
    
    
    
    print '----------------------'
    print '-------End------------'
    print '----------------------'