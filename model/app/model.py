# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 08:44:54 2021

@author: jgharris
"""
minLength=5
maxlines=8000000
#dataFile='/test/testshort.csv'
maxFeatures=1000
mindf=1
maxdf=1.0
mindf=1
testsize=.2
random_state=45
alphasmooth=1
ngramrange=(1,1)
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#import matplotlib
#import matplotlib.pyplot as plt
#import numpy as np
import pickle
import sys
class Model(object):
    def __init__(self):
        self.isloaded=False
        return

    def predict(self,features):
        return
    def fit (self,x,y):
        return  

class DocClf(Model):
    def __init__(self):
        return
    def fit(self,x,y):
            # generate dictionary of words and numb of word occurences
    # in each document
        self.vectorizer=\
        CountVectorizer(max_df=maxdf,min_df=mindf,max_features=maxFeatures,
                               ngram_range=ngramrange)
        
        xv=self.vectorizer.fit_transform(x)
        self.nbclf=MultinomialNB(alpha=alphasmooth)
        self.nbclf.fit(xv,y)
        ytrain=self.nbclf.predict(xv)
        return ytrain
    
    #predict for a group of x value
    def predict(self,x):
        if (len(x[0])<minLength):
            y=["No input"]
            return y
        try:
            xv=self.vectorizer.transform(x)
            y=self.nbclf.predict(xv)
        except:
            raise
        return y
    
    # Compute confidence given predicted values & return confusion matrix
    def confidence(self,ytest,ytestpred):
        conf_mat = confusion_matrix(ytest, ytestpred)
    # compute accuracy given predicted value
        labels = sorted(set(ytest))
        self.confidence=dict(zip(labels, conf_mat.diagonal()/
                                 conf_mat.sum(axis=0)))
        return conf_mat
    # get the Confidence score for a single item:
    def getConfidence(self,x,y):
        try:
            return self.confidence[y]
        except:
            return -1.0;
        

        
    
        

   
   