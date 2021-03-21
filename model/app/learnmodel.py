# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:55:54 2021

@author: jgharris
"""
root='C:/Users/jgharris/DocClass/'

dataFile='/data/shuffled-full-set-hashed.csv'
modelName="v2"


import statistics as stat
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from model import *
maxlines=800000
#dataFile='/test/testshort.csv'
maxFeatures=1000
mindf=1
maxdf=1.0
mindf=1
testsize=.2
random_state=45
alphasmooth=1
ngramrange=(1,1)
class documents:
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
    
def main():  
     # Set up corpus for training               
    corpus=documents()
    corpus.readFromFile(root+dataFile,maxline=maxlines)
    model1=DocClf()
    # split into test and training sets
    xtrain,xtest,ytrain,ytest=\
        train_test_split(corpus.words,corpus.y,test_size=testsize)
    ytrainpred=model1.fit(xtrain,ytrain)
    ytestpred=model1.predict(xtest)

    trainAccuracy=accuracy_score(ytrain,ytrainpred)
    testAccuracy=accuracy_score(ytest,ytestpred)
    controlAccuracy=accuracy_score(np.random.permutation(ytest),ytestpred)
    
    print('train=%6.2f  test=%6.2f control=%6.2f' % 
          (trainAccuracy,testAccuracy,controlAccuracy))
    conf_mat =model1.confidence(ytest, ytestpred)
    # compute accuracy given predicted value
   
    '''
    pickle.dump(vectorizer,open(root+'/test/'+modelName+"_vectors.pck","wb"))
    pickle.dump(nbclf,open(root+'/test/'+modelName+"_nbayes.pck","wb"))
    pickle.dump(confidence,open(root+'/test/'+modelName+"_conf.pck","wb"))
    '''
    pickle.dump(model1,open(root+modelName+".pckmdl","wb"))
    print(model1.confidence)
def docpeek():
    corpus=documents()
    corpus.readFromFile(root+dataFile,maxline=maxlines)
    print([(i,corpus.y.count(i)) for i in corpus.classes])
    corpus.makeDataFrame()
    x1=corpus.df[['class','count']].groupby(by='class')
    a1=x1.min()
    a2=x1.max()
    a3=x1.mean()
    a4=x1.std()
    a1.columns=['min']
    a2.columns=['max']
    a3.columns=['mean']
    a4.columns=['std']
    q=a1.merge(a2,left_index=True,right_index=True).merge(a3,left_index=True,right_index=True)
    q=q.merge(a4,left_index=True,right_index=True)
    return corpus,q
if  __name__=='__main__':
    main()
    

            
            
            

        
