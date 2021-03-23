# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 20:32:10 2021

@author: jgharris
"""

minLength=5
maxFeatures=1000
mindf=1
maxdf=1.0
mindf=1
alphasmooth=1
ngramrange=(1,1)
MAXSTRINGLENGH=7060
FIRSTSTRINGLENGTH=80
 
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import ComplementNB
#from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#import matplotlib
#import matplotlib.pyplot as plt
#import numpy as np

class DocClfComplNB():
    def __init__(self,maxStringLength=MAXSTRINGLENGH, \
                 firstStringLength=FIRSTSTRINGLENGTH):
        self.maxStringLength=maxStringLength
        self.firstStringLength=firstStringLength
        self.message="Naive Bayes Complement with "+"%5d" % maxFeatures + " features " + \
        " ngram-range "+"%2d" % ngramrange[0]+" to "+"%2d" % ngramrange[1] + \
        " maxString Length "+ "%6d" % self.maxStringLength
       
        return
    def preprocess(self,x):
        xprocessed=[]
        xbegin=[]
        for item in x:
            xprocessed.append(item[0:self.maxStringLength])
            xbegin.append(item[0:self.firstStringLength])
        return xprocessed,xbegin
    def fit(self,x,y):
            # generate dictionary of words and numb of word occurences
    # in each document
        xprocessed,xbegin=self.preprocess(x)
        self.vectorizer=\
        CountVectorizer(max_df=maxdf,min_df=mindf,max_features=maxFeatures,
                               ngram_range=ngramrange)
        xv=self.vectorizer.fit_transform(xprocessed)
        self.nbclf=ComplementNB(alpha=alphasmooth)
        self.nbclf.fit(xv,y)
        ytrain=self.nbclf.predict(xv)
        return ytrain
    
    #predict for a group of x value
    def predict(self,x):
        if (len(x[0])<minLength):
            y=["No input"]
            return y
        try:
            xprocessed,xbegin=self.preprocess(x)
            xv=self.vectorizer.transform(xprocessed)
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