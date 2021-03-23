# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:55:54 2021

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
from DocClf2 import DocClf2
from Documents import Documents
modelName="v3"

#dataFile='/test/testshort.csv'


maxlines=8000000
testsize=.2
random_state=45
MAXSTRINGLENGH=4000
FIRSTSTRINGLENGTH=80
   
def main():  
     # Set up corpus for training               
    corpus=Documents()
    corpus.readFromFile(root+dataFile,maxline=maxlines)
    ''' 
    model1=DocClfComplNB(maxStringLength=MAXSTRINGLENGH, \
                 firstStringLength=FIRSTSTRINGLENGTH)
        '''
    model1=DocClf2(maxStringLength=MAXSTRINGLENGH, \
                 firstStringLength=FIRSTSTRINGLENGTH)
    print()
    # split into test and training sets
    xtrain,xtest,ytrain,ytest=\
        train_test_split(corpus.words,corpus.y,test_size=testsize, \
                         random_state=random_state)
    ytrainpred=model1.fit(xtrain,ytrain)
    ytestpred=model1.predict(xtest)


    print([(i,ytest.count(i)) for i in sorted(set(ytest))])
    
    
    trainAccuracy=accuracy_score(ytrain,ytrainpred)
    testAccuracy=accuracy_score(ytest,ytestpred)
    controlAccuracy=accuracy_score(np.random.permutation(ytest),ytestpred)
    
    
    global conf_mat
    conf_mat =model1.confidence(ytest, ytestpred)
    print(model1.confidence)
    print()
    print( np.unique(ytestpred,return_counts=True))
    print()
    
    [print("%-25s" % key +" %5.3f" % value) for key,value in model1.confidence.items()]
    
    labels=[]
    [labels.append(key) for key in model1.confidence.keys()]
    for row in range(0,conf_mat.shape[0]):
       print( [" %4d" % conf_mat[row,col] for col in range(0,conf_mat.shape[1])])
    
    rowsum=conf_mat.sum(axis=0)
    colsum=conf_mat.sum(axis=1)
    print("item     rowsum      colsum")
    for ic in range(0,conf_mat.shape[0]):
        print("%-25s" % labels[ic] + " %5d" % rowsum[ic]+ " %5d" % colsum[ic])
      
    print("")
    print('train=%6.2f  test=%6.2f control=%6.2f' % 
          (trainAccuracy,testAccuracy,controlAccuracy))
 
    pickle.dump(model1,open(root+modelName+".pckmdl","wb"))
    print(model1.confidence)
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
def docpeek():
    corpus=Documents()
    corpus.readFromFile(root+dataFile,maxline=maxlines)
    print([(i,corpus.y.count(i)) for i in corpus.classes])
    corpus.makeDataFrame()
    x1=corpus.df[['class','count']].groupby(by='class')
    cnt=x1.count()
    a1=x1.min()
    a2=x1.max()
    a3=x1.mean()
    a4=x1.std()
    cnt.columns=['count']
    a1.columns=['min']
    a2.columns=['max']
    a3.columns=['mean']
    a4.columns=['std']
    q=cnt.merge(a1,left_index=True,right_index=True)\
        .merge(a2,left_index=True,right_index=True)\
        .merge(a3,left_index=True,right_index=True)
    q=q.merge(a4,left_index=True,right_index=True)
    return corpus,q
if  __name__=='__main__':
    main()
    

            
            
            

        
