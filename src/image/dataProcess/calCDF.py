# -*- coding:utf-8 -*-
'''
@author: Huaigui Ren
Created on : 2015年12月31日
time: 上午10:41:25
Function: 
'''
from numpy import *
from sklearn.datasets import load_svmlight_file
from matplotlib.font_manager import FontProperties
from pylab import *
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14) 


def pltCDF(data):
    from sklearn import preprocessing
    min_max_scaler =preprocessing.MinMaxScaler()
#     l = min_max_scaler.fit_transform(data) #取出数据
    l= data
    t = l[argsort(l)]  #排序
    print len(t)
    x = t/t.sum()  #计算概率值
    u = ones(len(t))
    m = u/u.sum()
    b = cumsum(m)
    l = cumsum(x) #累计和
    print l
    print b
    plot(t,l,label='data')
    plot(t,b,label='user')
#     plt.xlim(0,0.1)
    plt.show()

def GenderCDF(data,label,featureList):
    [m,n] = shape(data)
    cdfdata = data.toarray()
    maledata = [] #存放男性数据
    famaledata = [] #存放女性数据
    
    #print data.T[0]
#     l = cdfdata.T[1] #取出数据
#     print l
#     t = l[argsort(l)]  #排序
#     print t
#     x = t/t.sum()  #计算概率值
#     print x
#     l = cumsum(x) #累计和
#     print l
#     plot(t,l)
#     show()
    
    print len(label)
    #分开男女属性的数据
    for i in range(0,len(label)):
        if label[i] == 0.0:
            famaledata.append(cdfdata[i])
        if label[i] == 1.0:
            maledata.append(cdfdata[i])
    maledata = array(maledata)
    famaledata = array(famaledata)        

    
    for i in range(0,len(featureList)):
        l = maledata.T[i] #取出数据
        t = l[argsort(l)]  #排序
        x = t/t.sum()  #计算概率值
        l = cumsum(x) #累计和
        plot(t,l,label='male')
          
        fl = famaledata.T[i] #取出数据
        ft = fl[argsort(fl)]  #排序
        fx = ft/ft.sum()  #计算概率值
        fl = cumsum(fx) #累计和
        plot(ft,fl,'r--',label='famael')
        ylabel(r'$\rm{CDF} \%$', fontsize=16)
        xlabel(r'$\rm{feature}$', fontsize=16)
        title(featureList[i],fontproperties=font)
        plt.legend(loc=2)
        savefig("./picture/"+featureList[i]+".png")
        plt.close()
    #savefig("./picture/all.png")

    
    
if __name__ == '__main__':  
#     imagepath = "E:/Desktop/Image/SVMData/sina/sina_gender_libsvmdata_scale.txt"
#     userpath = "E:/Desktop/Image/SVMData/sina/userGenderLibsvmdata.txt"
#     contentpath = "E:/Desktop/Image/SVMData/sina/contentGenderLibsvmdata.txt"
#     data, label = load_svmlight_file(imagepath)
#     labels={}.fromkeys(label).keys() #获取标签类别数
#    
#     featureList=[u'HSV颜色空间h通道一阶矩',u'HSV颜色空间h通道二阶矩',u'HSV颜色空间h通道三阶矩',u'HSV颜色空间s通道一阶矩',u'HSV颜色空间s通道二阶矩',u'HSV颜色空间s通道三阶矩',u'HSV颜色空间v通道一阶矩',u'HSV颜色空间v通道二阶矩',u'HSV颜色空间v通道三阶矩',u'HSV颜色空间h通道均值',u'HSV颜色空间s通道均值',u'HSV颜色空间v通道均值',u'HSV颜色空间图片中心区域h通道均值',u'HSV颜色空间图片中心区域s通道均值',u'HSV颜色空间图片中心区域v通道均值',u'纹理0°熵',u'纹理0°能量',u'纹理0°对比度',u'纹理0°逆值差',u'纹理45°熵',u'纹理45°能量',u'纹理45°对比度',u'纹理45°逆值差',u'纹理90°熵',u'纹理90°能量',u'纹理90°对比度',u'纹理90°逆值差',u'纹理90°熵',u'纹理90°能量',u'纹理90°对比度',u'纹理90°逆值差',u'一维灰度众数',u'一维灰度熵',u'一维灰度能量',u'一维灰度均值',u'一维灰度标准差',u'一维灰度偏态',u'一维灰度峰态（峰值）',u'二维灰度熵',u'二维灰度能量（角二阶矩）',u'二维灰度对比度（主对角线惯性矩）',u'二维灰度逆值差']
#     UserStaticsList =['followNum','fansNum','pulishNum']  
#     contentStaticsList = [u'图片转发数',u'图片评论数',u'图片点赞数',u'图片发图时间',u'图片的数量'] 
#    
#     GenderCDF(data,label,featureList)
    print "END!"



