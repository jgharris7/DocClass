# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 08:44:54 2021

@author: jgharris
"""
minLength=5
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix
#import matplotlib
#import matplotlib.pyplot as plt
#import numpy as np
import pickle
import sys
class Model(object):
    def __init__(self):
        self.isloaded=False
        return
    def load(self,name):
        return
    def predict(self,features):
        return

class DocClf(Model):
    def __init__(self):
        return
    def load(self,modelName):
        try:
            self.vec=pickle.load(open(modelName+"_vectors.pck","rb"))
        except:
            err = "vectors "+str(sys.exc_info()[0])
            return err
        try:
            self.nbclf=pickle.load(open(modelName+"_nbayes.pck","rb"))
        except:
            err = "classifier "+str(sys.exc_info()[0])
            return err
        try: 
             self.conf=pickle.load(open(modelName+"_conf.pck","rb"))
        except:
             err = "confidence "+str(sys.exc_info()[0])
             return err
        self.isloaded=True
        return None
    
    def predictOne(self,words):
        if (not(self.isloaded)):
            return "No_model_loaded"
        if(len(words)<minLength):
            return "String_to_short_for_model"
        try:
            x1=self.vec.transform([words])
        except:
            return "failed_vectorizer_"+str(sys.exc_info()[0])
        try:
            result=self.nbclf.predict(x1)[0]
        except:
            return "failed_nbclf_"+str(sys.exc_info()[0])
        try:
            confidence=self.conf[result]
        except:
            return "failed_confidence_"+str(sys.exc_info()[0])
        return (result,confidence)
        

   
   