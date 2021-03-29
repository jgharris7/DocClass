# -*- coding: utf-8 -*-
"""
The version uses Cross validation to pick the "C" (regularization) parameter
 to learn the linear SVC model. Set the list of regularization paramaters
 cvec to the values to test. The routine will build the final model from the
 best performers
 it will take  a while to run
Created on Mon Mar 22 23:24:47 2021

@author: jgharris
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 23:01:34 2021

@author: jgharris
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:43:22 2021

@author: jgharris
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:09:34 2021

@author: jgharris
"""

root='C:/Users/jgharris/DocClass/'
outdir=root+'model/'

dataFile='/data/shuffled-full-set-hashed.csv'



import statistics as stat
import pandas as pd
import time
from sklearn.model_selection import train_test_split
 
from sklearn.metrics import accuracy_score
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from DocClfTLinSVC import DocClfTLinSVC
from Documents import Documents


#dataFile='/test/testshort.csv'
MAXFEATURES=1000

PENALTY='l2'
LOSS='squared_hinge'
DUAL=False
modelName="linSVCv2test"
os.environ['model_name']=modelName

maxlines=800000
testsize=.2
random_state=45
MAXSTRINGLENGH=7000
FIRSTSTRINGLENGTH=80
conf_mat=[]
cvec=[.5, .7,.9, 1.0, 1.1,1.2,1.3,1.4]
start_time=time.time()

def main():  
     # Set up corpus for training               
    corpus=Documents()
    corpus.readFromFile(root+dataFile,maxline=maxlines)
    ''' 
    model1=DocClfComplNB(maxStringLength=MAXSTRINGLENGH, \
                 firstStringLength=FIRSTSTRINGLENGTH)
        '''
    model1=DocClfTLinSVC(maxStringLength=MAXSTRINGLENGH, \
                 firstStringLength=FIRSTSTRINGLENGTH,
                 penalty=PENALTY,loss=LOSS,dual=DUAL,
                 maxFeatures=MAXFEATURES
                 )
    print()
    
    # split into test and training sets
    xtrain,xtest,ytrain,ytest=\
        train_test_split(corpus.words,corpus.y,test_size=testsize, \
                         random_state=random_state)
    scorelist=model1.crossVal(cvec,xtrain,ytrain)
    ibest=0
    mincv=9.e9
    for item in range(0,len(cvec)):
        meancvscore=scorelist[item].mean()
        if meancvscore<mincv:
            mincv=meancvscore
            ibest=item
        print(item,cvec[item],meancvscore,ibest)
    model1=DocClfTLinSVC(maxStringLength=MAXSTRINGLENGH, \
                 firstStringLength=FIRSTSTRINGLENGTH,
                 penalty=PENALTY,loss=LOSS,dual=DUAL,
                 maxFeatures=MAXFEATURES,creg=cvec[ibest]
                 )
    
            
    ytrainpred=model1.fit(xtrain,ytrain)
    ytestpred=model1.predict(xtest)

    trainAccuracy=accuracy_score(ytrain,ytrainpred)
    testAccuracy=accuracy_score(ytest,ytestpred)
    controlAccuracy=accuracy_score(np.random.permutation(ytest),ytestpred)
    
    
    global conf_mat
    conf_mat =model1.confidence(ytest, ytestpred)
    print(model1.confidence)
    print()
    print( np.unique(ytestpred,return_counts=True))
    print()
    
    [print("%-20s" % key +" %5.3f" % value) for key,value in model1.confidence.items()]
    for row in range(0,conf_mat.shape[0]):
       print( [" %4d" % conf_mat[row,col] for col in range(0,conf_mat.shape[1])])
    
    rowsum=conf_mat.sum(axis=0)
    colsum=conf_mat.sum(axis=1)
    labels=[]
    [labels.append(key) for key in model1.confidence.keys()]
    print("item     rowsum      colsum")
    for ic in range(0,conf_mat.shape[0]):
        print("%-25s" % labels[ic] + " %5d" % rowsum[ic]+ " %5d" % colsum[ic])
      
    print("")
    print('train=%6.2f  test=%6.2f control=%6.2f' % 
          (trainAccuracy,testAccuracy,controlAccuracy))
    # compute accuracy given predicted value
   
 
    pickle.dump(model1,open(outdir+modelName+".pckmdl","wb"))
    
    print(ytestpred[0])
    print(xtest[0][0:20])
    testfile=open(outdir+modelName+"testdata.txt","wt")
    
    testfile.write(ytestpred[0])
    testfile.write(",")
    testfile.write(xtest[0])
    testfile.write("\n")
    testfile.write(ytestpred[10])
    testfile.write(",")
    testfile.write(xtest[10])
    testfile.write("\n")
    testfile.write(ytestpred[25])
    testfile.write(",")
    testfile.write(xtest[25])
    testfile.write("\n")
    testfile.close()
    print( model1.message)
     
    
    
if  __name__=='__main__':
    main()
    
