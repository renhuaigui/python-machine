# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2016年4月5日
time: 下午7:51:10
Function: 
'''
import cv2
import numpy as np  
#import pdb  
#pdb.set_trace()#turn on the pdb prompt  
  
#read image  
def sift_extract(path):
    img = cv2.imread(path,cv2.IMREAD_COLOR)  
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    #cv2.imshow('origin',img);  
  
    #SIFT  
    detector = cv2.SIFT()  
    keypoints,descriptors = detector.detectAndCompute(gray,None)  
    
    #print descriptors
    #img = cv2.drawKeypoints(gray,keypoints)  
    #img = cv2.drawKeypoints(gray,keypoints,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  
    #cv2.imshow('test',img);  
    #cv2.waitKey(0)  
    #cv2.destroyAllWindows() 
    return descriptors


'''    
if __name__ == '__main__':
    path = 'E:/Desktop/Image/Original/qq/774395172-20150406-15.jpg'
    sift_extract(path)
'''
        