

#v2 implements the fixed validation dataset approach and fixes numpy seed
# learning curve using MLP for text classification

# 

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import roc_curve
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout
import time
import os
from numpy.random import seed
seed(1)

from sklearn.metrics import precision_recall_curve,average_precision_score
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))

    def plot_epoch_loss(history): # function to plot loss and accuracy over epochs
        
        loss = history.history['loss']
        val_loss = history.history['val_loss']   
        epochs = range(1, len(loss) + 1)    
        plt.clf()    
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()    
        pn=ofp_p+r'\\'+ofn_p+"TVLoss_"+str(X_train.shape[0])+".png"    
        plt.savefig(pn)    
        plt.close()  
        
        plt.clf()    
        acc = history.history['acc']
        val_acc = history.history['val_acc']    
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()    
        pn=ofp_p+r'\\'+ofn_p+"TVAccuracy_"+str(X_train.shape[0])+".png"    
        plt.savefig(pn)    
        plt.close() 
        
        return()
    
        
    def dltc(): # function to perform deep learning for text classification
        
        global X_train,Y_train,X_test,Y_test,X_val,Y_val
        start = time.time()  
        
        
        ## DEFINE MODEL ##
        
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.25))
        #model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
    
       
        batch_size = bsize # max of 5, twenty iterations per training epoch
        epochs = epoch # number of passes through training data  
    
            
        Y_train =np.array(Y_train).astype(int)
        Y_val = np.array(Y_val).astype(int)
    
        history=model.fit(X_train, Y_train,
        	                    batch_size=batch_size,
        	                    nb_epoch=epochs,
        	                    verbose=1,
        	                    validation_data=(X_val, Y_val));
    
        plot_epoch_loss(history) # plot loss curve
    
        prob_val=model.predict(X_val, verbose=0)  # make prob predictions on unclassified data
        prob_val=[i for i in prob_val] # get prob relevance
        tr_val=[float(i) for i in Y_val] # true label for val data as float    
        fpr, tpr, thresholds =roc_curve(tr_val,prob_val)
        optco=thresholds[np.argmax(tpr>0.95)] # optimal cutoff
        pred_val=[1. if i>optco else 0. for i in prob_val]# pred is 1 if prob>optimal cutoff as determined from val data       
        pf1=metrics.f1_score(tr_val,pred_val) # predicted f1        
        ppr=metrics.precision_score(tr_val,pred_val) # predicted  precision
        prec=metrics.recall_score(tr_val,pred_val) # predicted  recall      
        proc=metrics.roc_auc_score(tr_val,prob_val) # predicted roc auc measured on test data   
        precision, recall, pr_thresholds = precision_recall_curve(tr_val, prob_val)
    #    p_prre_auc= metrics.auc(recall,precision,reorder=True) # pr auc   
        p_prre_auc= metrics.average_precision_score(tr_val,prob_val) # pr auc
        
        
        # Make predictions on unclassified data
        
        prob_test=model.predict(X_test, verbose=0)  # make prob predictions on unclassified (test) data
        tr_test=Y_test# target column (true label) as vector
        tr_test=[float(i) for i in tr_test] # true label as float    
        fpr, tpr, thresholds =roc_curve(tr_test,prob_test)
        optco=thresholds[np.argmax(tpr>0.95)] # optimal cutoff
        pred_test=[1. if i>optco else 0. for i in prob_test]# pred is 1 if prob>0.5       
        af1=metrics.f1_score(tr_test,pred_test) #actual f1        
        apr=metrics.precision_score(tr_test,pred_test) #actual precision
        arec=metrics.recall_score(tr_test,pred_test) #actual recall      
        aroc=metrics.roc_auc_score(tr_test,prob_test) #actual roc auc   
        precision, recall, a_thresholds = precision_recall_curve(tr_test, prob_test)
    #    a_prre_auc= metrics.auc(recall,precision,reorder=True) # pr auc
        a_prre_auc= metrics.average_precision_score(tr_test,prob_test) # pr auc
    
        ndata=X_train.shape[0]    
    
        t=(time.time() - start)/3600. # time taken in hours
    
        r = [ndata,t,pf1, af1, ppr,apr, prec,arec,proc,aroc,p_prre_auc,a_prre_auc] # list of results for output
            
        print r
                   
        return(r)
    
    ## USER INPUTS#####
    
    ifp=path  # Filepath for all files (input and output)
    ifn=r"\Data\As_All.csv" # Input filename
    txtcol=2 # text column zero index in training data
    targcol=4 # target column zero index in training data 
    ngram=5 # ngram length
    stop_words='english' # can add to default stopwords if needed
    csize=500 # chunk size for simulation
    vsize=500 # val data size for making performance predictions
    tsize=2500 # test data size for verifying performance predictions
    bsize=128 # minimum batch size (samples used for training in each pass)
    epoch=10 # number of passes through data
    
    ver="v2_MoA" # version
    
    
    ### DERIVED INPUTS ####
    
    ifpn=ifp+ifn  # input filepath and name
    
    df = pd.read_csv(ifpn, header=0) # read simulation data as dataframe
    df=df.sample(frac=1,replace=False,random_state=45) # reshuffle with fixed seed
    print (df.head(3))
    
    header = list(df.columns.values) # data header
    targcolhd=header[targcol] # header of target column 
    txtcolhd=header[txtcol] # text column header 
    
    ofp=path+r"\Output-MLP" # output file path
    if not os.path.isdir(ofp):
        os.makedirs(ofp)    
    ofn=r"MLP_LearningCurve_"+ver+r".csv" # output filename
    ofpn=ofp+r'\\'+ofn # output filepath and filename
    
    ofp_p=ofp # output file path
    ofn_p=r'MLP_' # output file name
    
    
    
    df_text = df[txtcolhd] # text vector from training dataframe        
    df_text=df_text.tolist() # text vector as list
    txt = [x.decode('ascii', 'ignore') if isinstance(x,basestring) else str(x) for x in df_text] # convert to ascii       
    df_target = df[targcolhd].values # target values as vector               
    
    dfp = pd.DataFrame({'targ':df_target})
    dfpseries = pd.Series(dfp['targ'])
    dfpseries_counts = dfpseries.value_counts()    
    l_dfpseries_counts = len(dfpseries_counts)
    
    if l_dfpseries_counts == 2:
        target= df_target.astype(np.float)    
    else:
        target= df_target
    
    vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, ngram), max_df=0.5, stop_words=stop_words) # define vectorizer 
    X= vectorizer.fit_transform(txt) # vectorize training data text  
    Y=target
    
    X_val=X[0:vsize,:] # select validation data
    Y_val=Y[0:vsize]
    Y_val = np.array(Y_val).astype(int);
    
    X_test=X[vsize:tsize+vsize,:] # select test data
    Y_test=Y[vsize:tsize+vsize]
    Y_test = np.array(Y_test).astype(int);
    
    
    print('Shape of X_test:', X_test.shape)
    print('Shape of X_val:', X_val.shape)
    
    X_r=X[tsize+vsize:,:] # residual X to select training data from in batches of incremental chunk size csize
    Y_r=Y[tsize+vsize:] # residual Y
    
    
    finres=[] # initialize final results list 
    
    for i in range(1,int(X_r.shape[0])/csize+1): # loop over chunks
        X_train=X_r[0:i*csize,:]
        Y_train=Y_r[0:i*csize]
        finres.append(dltc()) # call dltc and append results
    
    op = map(None, finres)    # output list
    print (op) 
    
    header_pr = ["n_Training","RunTime-hrs","Predicted F1-score", "Actual F1-score","Predicted Precision", "Actual Precision", "Predicted Recall","Actual Recall","Predicted ROC-AUC","Actual ROC-AUC","Predicted PrRec-AUC","Actual PrRec-AUC"]    
    
    with open(ofpn, "w") as output: # Output results          
        writer = csv.writer(output, lineterminator='\n')    
        writer.writerow(header_pr)        
        for row in op:
            tempRow = row
            writer.writerow(tempRow)
