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

dataFile='/data/shuffled-full-set-hashed.csv'



import statistics as stat
import pandas as pd
 
from sklearn.model_selection import train_test_split
 
from sklearn.metrics import accuracy_score
 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from DocClfTSVC import DocClfTSVC
from Documents import Documents


#dataFile='/test/testshort.csv'
MAXFEATURES=1000
MAXSTRINGLENGH=7060
FIRSTSTRINGLENGTH=80
GAMMA=1/MAXFEATURES
KERNEL="poly"
COEF0=0.0
DEGREE=3
SHRINKING=True
modelName="polySVCv0"


maxlines=80000000
testsize=.3
random_state=45
MAXSTRINGLENGH=4000
FIRSTSTRINGLENGTH=80
conf_mat=[]
def main():  
     # Set up corpus for training               
    corpus=Documents()
    corpus.readFromFile(root+dataFile,maxline=maxlines)
    ''' 
    model1=DocClfComplNB(maxStringLength=MAXSTRINGLENGH, \
                 firstStringLength=FIRSTSTRINGLENGTH)
        '''
    model1=DocClfTSVC(maxStringLength=MAXSTRINGLENGH, \
                 firstStringLength=FIRSTSTRINGLENGTH, \
                 kernel=KERNEL,degree=DEGREE,\
                 gamma=GAMMA,\
                 coef0=COEF0,\
                 shrinking=SHRINKING,\
                 maxFeatures=MAXFEATURES\
                 )
    print (model1.message)
    
    print()
    # split into test and training sets
    xtrain,xtest,ytrain,ytest=\
        train_test_split(corpus.words,corpus.y,test_size=testsize, \
                         random_state=random_state)
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
   
 
    pickle.dump(model1,open(root+modelName+".pckmdl","wb"))
    
    print(ytestpred[0])
    print(xtest[0][0:20])
    testfile=open(root+modelName+"testdata.txt","wt")
    
    testfile.write(ytestpred[0])
    testfile.write("\n")
    testfile.write(xtest[0])
    testfile.write("\n")
    testfile.write(ytestpred[10])
    testfile.write("\n")
    testfile.write(xtest[10])
    testfile.write("\n")
    testfile.close()
    print( model1.message)
     
    
    
if  __name__=='__main__':
    main()
    