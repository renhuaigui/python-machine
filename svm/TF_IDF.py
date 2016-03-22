#coding:utf-8
'''
@author: Huaigui Ren
Created on : 2015年9月11日
time: 1:07:39
Function: 
'''

import os
import math
import numpy as np

def savedata(tfidf,y,numOfAttr,tfidf_File = 'tfidfData.base'):
    """
    save from sparse dict,return tf_idf_datax
    """
    def sparseToFull(sparseList,numOfAttr):
        """
        input a line ([(i,e),(i,e)])
        output a line ([0.1,0,0,0,0.3,0])
        """
        tmp_dict = dict()
        for (i,e) in sparseList: tmp_dict.setdefault(i,e)
        full_list = list()
        for i in range(numOfAttr):
            if tmp_dict.has_key(i):
                full_list.append(tmp_dict[i])
            else:
                full_list.append(0)
            
        return full_list        
    datax = list()    
    with open(tfidf_File,'w') as fw:
        for idx in range(len(tfidf)):
            full_list = sparseToFull(tfidf[idx],numOfAttr)
            datax.append(full_list)
            fw.write(str(full_list).replace('[','').replace(']','') +str(y[idx]) + '\n')
    return datax
            
    

class tf_idf():
    """
    tf-idf 
    The algorithm is described in
        http://en.wikipedia.org/wiki/Tf-idf
        
    """
    def __init__(self,X,sparse=False):
        """
        if NOT sparse: X = [[1,0,0,2],[1,0,1,0]]
        if sparse:     X = [[(iid1,times1),(iid2,times2)],[(iid3,times3)]]
        """
        self.Row,self.Col = np.array(X).shape
        self.X = list()
        self.freq_X = list()
        if sparse:
            self.X = X
        else:    
            self.X = self.to_sparse(X)
             
        
    def to_sparse(self,X):
        """
        trans a raw data ( X = [[1,0,0,1],[1,0,1,0]]) into sparse
        """
        sX = list()
        for uid in range(len(X)):
            each = X[uid]
            tmp_ulist = list()
            for iid in range(len(each)):
                e_value = each[iid]
                if e_value:
                    tmp_ulist.append((iid,e_value))
                    
            sX.append(tmp_ulist)        
                    
        return sX
    
    #tf:  term frequency 
    #perform on sparse X
    def raw_frequency(self):
        for each in self.X:
            tmp_sum = sum([e[-1] for e in each])
            tmp_freq = [(i,float(e)/tmp_sum) for (i,e) in each]
            self.freq_X.append(tmp_freq)
            
    def boolean_frequency(self):
        for each in self.X:
            self.freq_X.append([(i,1) for (i,e) in each])
            
    def logarithmically_scaled_frequency(self):
        for each in self.X:
            tmp_sum = sum([e[-1] for e in each])
            tmp_freq = [(i,math.log(float(e)/tmp_sum)+1) for (i,e) in each ]
            self.freq_X.append(tmp_freq)
            
    def augmented_frequency(self):
        for each in self.X:
            tmp_sum = sum([e[-1] for e in each])
            tmp_freq = [float(e[1])/tmp_sum for e in each ]
            max_freq = max(tmp_freq)
            self.freq_X.append([(i,0.5+0.5*tmp_freq[i]/max_freq) for i in range(len(tmp_freq))])
            
            
    #idf:  inverse document frequency
    def idf_prepear(self):
        #|D|:  the total number of documents in the corpus
        self.D = self.Row
    
        #1+|{d-D:t-d}|:  number of documents where trem t appears
        #store in Ddict
        Ddict = dict()
        for each in self.X:
            for (e,v) in each:
                Ddict.setdefault(e,0.0)
                Ddict[e] += v
                
        return Ddict        
                
    def idf(self,term,Ddict):
        return math.log(self.D/1. + Ddict[term])
    
    
    
    def tfidf(self,tf_type=0):
        """
        choose tf type
        0 - raw_frequency
        1 - boolean_frequency
        2 - logarithmically_scaled_frequency
        3 - augmented_frequency
        
        """
        if tf_type == 0:
            self.raw_frequency()
        elif tf_type == 1:
            self.boolean_frequency()
        elif tf_type == 2:
            self.logarithmically_scaled_frequency()
        elif tf_type == 3:
            self.augmented_frequency()
        else:
            raise Exception("wrong tf_type !")

        
        Ddict = self.idf_prepear()        
        #clear space
        self.X = list()      
        
        #run tfidf
        self.tfidf = list()
        for each in self.freq_X:
            tmp_list = list()
            for (i,e) in each:
                tmp_list.append((i,e * math.log(Ddict[i])))
                
            self.tfidf.append(tmp_list)    
            
        return self.tfidf    
def TF_IDF(dataX,y,savefilepath):
    #获取文件路径和文件名
    dirpath = os.path.dirname(savefilepath)
    name = os.path.splitext((os.path.basename(savefilepath)))[0]
    #生成存储路径
    tfidf_File = os.path.join(dirpath,name+'_ifdifdata.txt')
    T = tf_idf(dataX)
    tfidf = T.tfidf(tf_type = 0)
    X = savedata(tfidf,y,T.Col,tfidf_File)
    return X

    
'''  
test 

if __name__ == '__main__':
    base =  r'..\data\result10.txt'
    TF_IDF(base,-3)
    #X,y = loaddata(Base)
    #print 'load completed !'
    #X = [[2,2],[0,3],[3,5]]
    #T = tf_idf(X)
    #print X
    #print T
    #print 'init finished !'
    #tfidf = T.tfidf(tf_type = 0)
    #print tfidf
    #print 'transTo tfidf done !'
    #savedata(tfidf,y,T.Col,tfidf_File = 'tfidfData.base')
    #print 'save end !'
'''