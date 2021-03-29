# -*- coding: utf-8 -*-
"""
Document classifier using Linear Support Vector Machine Classifier
preprocessed with TD-IDF vectorizer
Created on Mon Mar 22 20:26:18 2021

@author: jgharris
"""
# Preprocessing parameters
minLength=5             #Mininmun number of words for running classifer
MAXFEATURES=2000
mindf=1
maxdf=1.0
mindf=1
alphasmooth=1
ngramrange=(1,3)
MAXSTRINGLENGH=7060
FIRSTSTRINGLENGTH=80
CREG=1.

# SVC options
PENALTY='l2'
LOSS='squared_hinge'
DUAL=False

 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
#from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import unittest
import pickle
import time
 
#import matplotlib
#import matplotlib.pyplot as plt
#import numpy as np

# path is './' for final build, but may be '../' for testing in IDE
 
unit_temp_file_path='./'
unit_model_name='linSVCv0'
start_time=time.time()
class DocClfTLinSVC():
    def __init__(self,maxStringLength=MAXSTRINGLENGH, \
                 firstStringLength=FIRSTSTRINGLENGTH,
                 penalty=PENALTY,loss=LOSS,dual=DUAL,
                 creg=CREG,
                 maxFeatures=MAXFEATURES
                 ):
        self.maxStringLength=maxStringLength
        self.firstStringLength=firstStringLength
        self.loss=loss
        self.penalty=penalty
        self.creg=creg
        self.dual=dual
        self.maxFeatures=maxFeatures
        self.message="LinearSVC using TF-IDF with "+"%5d" % maxFeatures + " features " + \
        " ngram-range "+"%2d" % ngramrange[0]+" to "+"%2d" % ngramrange[1] + \
        " maxString Length "+ "%6d" % self.maxStringLength +"Creg="+"%4.2f" % self.creg
       
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
        self.nbclf=LinearSVC(penalty=self.penalty,loss=self.loss,dual=self.dual,
                C=self.creg)
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
    def crossVal(self,cvec,xtrain,ytrain):
        scorelist=[]
        global start_time
        for item in cvec:
           modelt=make_pipeline(
            TfidfVectorizer(max_df=maxdf,min_df=mindf,max_features=self.maxFeatures,
                               ngram_range=ngramrange),
            LinearSVC(penalty=self.penalty,loss=self.loss,dual=self.dual,
                C=item)
            )
           scores=cross_val_score(modelt,xtrain,ytrain)
           print("time=%7.1f " % (time.time()-start_time),end="")
           print("%5.2f " % item, end=' ')
           [print("%10.5f " % xval,end='') for xval in scores]
           print("")
           scorelist.append(scores)
        return scorelist
           
           
        
    
    # Compute confidence given predicted values & return confusion matrix
    
    def confidence(self,ytest,ytestpred):
        conf_mat = confusion_matrix(ytest, ytestpred)
    # compute accuracy given predicted value
        labels = sorted(set(ytest))
        self.confidence=dict(zip(labels, conf_mat.diagonal()/
                                 (.1+conf_mat.sum(axis=0))))
        return conf_mat
    #
    # get the Confidence score for a single item:
    # "x" is stuck in here, but not used because it may be possible
    # to improve the Confidence estimate by metrics like
    # distance from separating boundary or a relative score
    #    
    def getConfidence(self,x,y):
        try:
            return self.confidence[y]
        except:
            return -1.0;
        
# Note that directory at top may need to be changed depending on
# whether this is run from the ./app or home directory
#  This tries to catch that error

class TestDocLinSVC(unittest.TestCase):
    def setUp(self):
       modelFile=unit_temp_file_path+unit_model_name+".pckmdl"
       try:
          with open(modelFile,"rb") as myFile:
             self.myModel=pickle.load(myFile)
          testF=open(unit_temp_file_path+unit_model_name+"testdata.txt","rt")
       except:
           with open('../'+modelFile,"rb") as myFile:
             self.myModel=pickle.load(myFile)
           testF=open('../'+unit_temp_file_path+unit_model_name+"testdata.txt","rt")
       
       self.cases=[]

       for line in testF:
           y,x=line.strip().split(',')
           self.cases.append((x,y))
       testF.close()
    def testpredict(self):
        print("")
        for case in self.cases:
            result=self.myModel.predict([case[0]])
            print((result,case[1]))
            self.assertEqual(result,case[1])
            
        
if __name__=='__main__':
    unittest.main()     
       
           

        
    
        

