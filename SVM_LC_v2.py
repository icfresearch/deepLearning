# -*- coding: utf-8 -*-
"""
v2 implements the fixed validation dataset approach and fixes numpy seed
SVM_v1 expands training set sequentially in the order of the original input file

v3 expands training set sequentially in the order of the original input file
v2 find youden maximum
For learning curve with ROC
Final_v1=Revised version of no bias correction with random sampling
@author: 13963

"""


import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import roc_curve
import time
import os
from numpy.random import seed
seed(1)

from sklearn.metrics import precision_recall_curve

path = r"C:\DeepLearning"
    
def mlfunc(txtcol, targcol,optmet,csize): # main ML function. .
        start = time.time()
        global tdf, vdf, trdf  # test, validation and training dataframes
        header=map(None,tdf.columns) # header      
        txtcolhd=header[txtcol]  # header of text col       
        targcolhd=header[targcol] # header of target col          
        df_text_tr = trdf[txtcolhd] # text vector from training dataframe        
        df_text_tr=df_text_tr.tolist() # text vector as list
        txt_tr = [x.decode('ascii', 'ignore') for x in df_text_tr if isinstance(x,basestring)] # convert to ascii       
        df_text_t = tdf[txtcolhd] # text vector from test dataframe        
        df_text_t=df_text_t.tolist() # as list        
        df_text_v = vdf[txtcolhd] # text vector from validation dataframe        
        df_text_v=df_text_v.tolist() # as list        
        txt_t = [x.decode('ascii', 'ignore') for x in df_text_t if isinstance(x,basestring)] # convert to ascii     
        txt_v = [x.decode('ascii', 'ignore') for x in df_text_v if isinstance(x,basestring)] # convert to ascii     

        df_target = trdf[targcolhd].values # target values as vector from training dataframe  
        ## updated on 6/30/2015 to fix a bug on binary cross-validation
        dfp = pd.DataFrame({'targ':df_target})
        dfpseries = pd.Series(dfp['targ'])
        dfpseries_counts = dfpseries.value_counts()    
        l_dfpseries_counts = len(dfpseries_counts)

        if l_dfpseries_counts == 2:
            target = df_target.astype(np.float)    
        else:
            target = df_target

        vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, ngram), max_df=0.5, stop_words=stop_words) # define vectorizer 
        X_train = vectorizer.fit_transform(txt_tr) # vectorize training data text  
        X_test = vectorizer.transform(txt_t) # vectorize validation data text
        X_val = vectorizer.transform(txt_v) # vectorize validation data text
        clf.fit(X_train, target)  # fit best classifier 
        pred_t = clf.predict(X_test) # make prediciton on test text
        pred_v = clf.predict(X_val) # make prediciton on validation text

        ndata=len(trdf) # size of unsampled training data  

        if hasattr(clf, "predict_proba"): # to support classifiers with probability scores
            prob_t = clf.predict_proba(X_test)[:, 1]            
            prob_v = clf.predict_proba(X_val)[:, 1]            
        else:  # use decision function for SVM and classifiers with no prob scores
            dist_t = clf.decision_function(X_test) # distance (+/- from the boundary)
            prob_t = (dist_t - dist_t.min()) / (dist_t.max() - dist_t.min()) # convert to normalized percentile score. 
            dist_v = clf.decision_function(X_val) # distance (+/- from the boundary)
            prob_v = (dist_v - dist_v.min()) / (dist_v.max() - dist_v.min()) # convert to normalized percentile score. 

        tr_v=vdf[targcolhd] # target column (true label) of the validation dataset
        tr_v=[float(i) for i in tr_v] # true label as float of the validation dataset
        fpr, tpr, thresholds =roc_curve(tr_v,prob_v)
        optco=thresholds[np.argmax(tpr>0.95)] # optimal cutoff
        pred_v=[1. if i>optco else 0. for i in prob_v]# pred is 1 if prob>optco      
        
        pf1=metrics.f1_score(tr_v,pred_v) # predicted f1 based on predictions to the validation dataset        
        ppr=metrics.precision_score(tr_v,pred_v) # predicted precision based on predictions to the validation dataset
        prec=metrics.recall_score(tr_v,pred_v) #predicted recall based on predictions to the validation dataset
        proc=metrics.roc_auc_score(tr_v,prob_v) #actual roc auc measured on validation data   
        precision, recall, pr_thresholds = precision_recall_curve(tr_v, prob_v)
        p_prre_auc= metrics.auc(precision, recall,reorder=True) # pr auc   

        prob_t=np.array(prob_t)  # convert to array      
        pred_t=[1. if i>optco else 0. for i in prob_t]# predicted label in the test data based on optimal prob cutoff previously determined from the val set       
        tr_t=tdf[targcolhd] # target column (true label) as vector
        tr_t=[float(i) for i in tr_t] # true label as float
        af1=metrics.f1_score(tr_t,pred_t) #actual f1 based on test dataset 
        apr=metrics.precision_score(tr_t,pred_t) #actual precision based on test dataset
        arec=metrics.recall_score(tr_t,pred_t) #actual recall based on test dataset
        aroc=metrics.roc_auc_score(tr_t,prob_t) #actual roc auc measured on test data   
        precision, recall, a_thresholds = precision_recall_curve(tr_t, prob_t)
        a_prre_auc= metrics.auc(precision, recall,reorder=True) # pr auc

        t=(time.time() - start)/3600. # time taken in hours
        r = [ndata,t,clf, ngram, pf1, af1, ppr,apr, prec,arec,proc,aroc,p_prre_auc,a_prre_auc] # list of results for output
        
        print r
               
        return(r)


###### Main Code ######

finres = [] # initialize list of final results
csize=500 # chunk size
vsize=500 # validation data size
tsize=2500 # test data size
ngram = 5 # ngram max
nfold = 5 # n fold cross validation
stop_words='english' # can add to default stopwords if needed

ifp=path  # Filepath for input file 
ifn=r"\Data\As_All.csv" # Input filename for training dataset: User Must Specify
txtcol=2 # text column zero index
targcol=4 # target column zero index
ifpn=ifp+ifn  # input filepath and name

df = pd.read_csv(ifpn, header=0) # read original data as dataframe
df=df.sample(frac=1,replace=False,random_state=45) # reshuffle with fixed seed
print (df.head(3))

header = list(df.columns.values) # original data header

ofp=path+r"\Output" # output file path
if not os.path.isdir(ofp):
    os.makedirs(ofp)
ver='v2_MoA' # version
ofn="SVM_LearningCurve_"+ver+r".csv" # output filename
ofpn=ofp+r'\\'+ofn # output filepath and filename

clf=LinearSVC(random_state=45) # single classifier chosen for this analysis
          
vdf=df.head(vsize) # carve out validation dataset. purpose of validation dataset is for predicted scores
nrow=df.shape[0]
df1=df.tail(nrow-vsize) # df1 is the dataset minus the validation dataset
tdf=df1.head(tsize) # carve out test dataset. purpose of test dataset is for actual scores
df1=df.tail(nrow-tsize-vsize) # df1 is the dataset minus the validation and test dataset

print('Shape of X_test:', tdf.shape)
print('Shape of X_val:', vdf.shape)


nrow=df1.shape[0] # num rows in residual training data

for i in range(1,len(df1)/csize+1): # loop over chunks
    trdf=df1.head(i*csize) # expand training dataset
    finres.append(mlfunc(txtcol,targcol,'f1',csize)) # call mlfunc and append results

op = map(None, finres)    # output list
print (op) 
csvfile = ofpn

header_pr = ["n_Training",r"Run Time (hrs)","Optimal Algorithm", "Optimal Word Grouping", "Predicted F1-score", "Actual F1-score","Predicted Precision", "Actual Precision", "Predicted Recall","Actual Recall","Predicted ROC-AUC","Actual ROC-AUC","Predicted PrRec-AUC","Actual PrRec-AUC"]

with open(csvfile, "w") as output: # Output results          
    writer = csv.writer(output, lineterminator='\n')    
    writer.writerow(header_pr)        
    for row in op:
        tempRow = row
        writer.writerow(tempRow)