# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:00:39 2021

@author: jgharris
"""

import pandas as pd
class Documents:
    # y = classification
    # words = words in document ("x" values)
    # counts = number of words in each document
    # classes =set of document classes
    
    def __init__(self):
        return
    
    def readFromFile(self,file,maxline=9.e99):
        self.y = []
        self.words=[]
        self.counts=[]
        i=0
        with open(file, "r") as infile:
            for line in infile:
                y,x=line.strip().split(',')
                self.y+=[y]
                self.words+=[x]
                self.counts+=[len(x.strip().split(' '))]
                i=i+1
                if(i>maxline): 
                   break
        self.classes=set(self.y)
    def makeDataFrame(self):
        self.df=pd.DataFrame([self.y,self.counts,self.words]).transpose()
        self.df.columns=['class','count','words']
        self.df['count']=pd.to_numeric(self.df['count'])
 