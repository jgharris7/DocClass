# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:48:41 2021

@author: jgharris
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:40:14 2021

@author: jgharris
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 20:26:18 2021

@author: jgharris
"""

## preprocessing (vectorizer) parameters)
minLength=5
MAXFEATURES=2000
mindf=1
maxdf=1.0
mindf=1
alphasmooth=1
ngramrange=(1,1)
MAXSTRINGLENGH=7060
FIRSTSTRINGLENGTH=80

# Machine learning parameters
GAMMA=1/MAXFEATURES
KERNEL="linear"
COEF0=0.0
DEGREE=3
SHRINKING=True

 
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
#from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#import matplotlib
#import matplotlib.pyplot as plt
#import numpy as np

class DocClfTSVC():
    def __init__(self,maxStringLength=MAXSTRINGLENGH, \
                 firstStringLength=FIRSTSTRINGLENGTH,
                 kernel=KERNEL,
                 degree=DEGREE,
                 gamma=GAMMA,
                 coef0=COEF0,
                 shrinking=SHRINKING,
                 maxFeatures=MAXFEATURES
                 ):
        self.maxStringLength=maxStringLength
        self.firstStringLength=firstStringLength
        self.kernel=kernel
        self.gamma=gamma
        self.coef0=coef0
        self.shrinking=shrinking
        self.maxFeatures=maxFeatures
# Message to document what model was implemented
        self.message="SVC using TF-IDF with "+"%5d" % maxFeatures + " features " + \
        " ngram-range "+"%2d" % ngramrange[0]+" to "+"%2d" % ngramrange[1] + \
        " maxString Length "+ "%6d" % self.maxStringLength+" "+self.kernel+" kernel"
       
        return

# Truncate string. Not implemented--possibility of combining counts from
#  begining of document with counts from most of document
# Documents must be truncated due to apparent length limit to GET URL

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
        TfidfVectorizer(max_df=maxdf,min_df=mindf,max_features=self.maxFeatures,
                               ngram_range=ngramrange)
        xv=self.vectorizer.fit_transform(xprocessed)
        self.nbclf=SVC(kernel=self.kernel, gamma=self.gamma,
                       coef0=self.coef0,shrinking=self.shrinking)
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
                                 (.1+conf_mat.sum(axis=0))))
        return conf_mat
    # get the Confidence score for a single item:
    def getConfidence(self,x,y):
        try:
            return self.confidence[y]
        except:
            return -1.0;        

        
    
        

