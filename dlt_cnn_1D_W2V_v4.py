# -*- coding: utf-8 -*-
"""

v4 implements the fixed validation dataset approach and fixes numpy seed
v3 introduces sequential expansion with fixed seed
v2 places embedding matrix outside loop to save time
This script loads pre-trained GloVe embeddings into a frozen Keras Embedding layer,
and uses it to train a text classification model  using 1D CNN
@author: 13963
"""

from __future__ import print_function
import time
import csv
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_curve
from gensim.models import Word2Vec
from numpy.random import seed
seed(1)

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

path = r"C:\DeepLearning"
    
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

def cnn_1D_fit():

    start_time = time.time()
    
    global X_val,X_train,X_test,w2vmodel,y_train,y_val,y_test   

    
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False) 

    
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)
    
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    
    history=model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs  ,
              validation_data=(X_val, y_val))

    plot_epoch_loss(history) # plot loss curve

    prob_val=model.predict(X_val, verbose=0)  # make prob predictions on val data (softmax style output)
    prob_val=[i[1] for i in prob_val] # get prob relevance
    tr_val=[float(i[1]) for i in y_val] # true label for val data as float    
    fpr, tpr, thresholds =roc_curve(tr_val,prob_val)
    optco=thresholds[np.argmax(tpr>0.95)] # optimal prob cutoff
    pred_val=[1. if i>optco else 0. for i in prob_val]# pred is 1 if prob>optimal cutoff as determined from validation data       
    pf1=metrics.f1_score(tr_val,pred_val) # predicted f1 based on val data        
    ppr=metrics.precision_score(tr_val,pred_val) # predicted precision
    prec=metrics.recall_score(tr_val,pred_val) # predicted recall      
    proc=metrics.roc_auc_score(tr_val,prob_val) # predicted roc auc measured on test data   
    precision, recall, pr_thresholds = precision_recall_curve(tr_val, prob_val)
    p_prre_auc= metrics.auc(precision, recall,reorder=True)    # pr auc
    
    prob_test=model.predict(X_test, verbose=0)  # make prob predictions on unclassified (test) data (softmax style output)
    prob_test=[i[1] for i in prob_test] # get prob relevance
    pred_test=[1. if i>optco else 0. for i in prob_test]# pred is 1 if prob on the test data > optimal cutoff  as determined from val data       
    tr_test=[float(i[1]) for i in y_test] # true label for test data as float  
    af1=metrics.f1_score(tr_test,pred_test) #actual f1 measured on test data       
    apr=metrics.precision_score(tr_test,pred_test) #actual precision measured on test data
    arec=metrics.recall_score(tr_test,pred_test) #actual recall measured on test data     
    aroc=metrics.roc_auc_score(tr_test,prob_test) #actual roc auc measured on test data   
    precision, recall, a_thresholds = precision_recall_curve(tr_test, prob_test)
    a_prre_auc= metrics.auc(precision, recall,reorder=True)# pr-auc
    ndata=X_train.shape[0]    
    
    t=(time.time() - start_time)/3600. # time taken in seconds
    
    r = [ndata,t,pf1, af1, ppr,apr, prec,arec,proc,aroc,p_prre_auc,a_prre_auc] # list of results for output        
       
    print("Time to run CNN model = --- %s hours ---" % ((time.time() - start_time)/3600.))
    print (r)

    return(r)



###### MAIN CODE #######

ver="W2V_v4_MoA"

MAX_SEQUENCE_LENGTH = 1000 # max length of document
MAX_NUM_WORDS = 20000 # vocab size
EMBEDDING_DIM = 1000 # len of embedding in W2V

#fpm=r'C:\W2V\Models' # Filepath to stored w2v models
fpm=path+r'\W2V\Models' # Filepath to stored w2v models
w2vmodel = Word2Vec.load(fpm+r'\abstracts_phrases_dl.model') # load w2v model
voc=w2vmodel.wv.vocab # vocabulary of model
voc=list(voc.keys()) # make list of vocab
print('Found %s unique tokens in W2V vocab.' % len(voc))

#ifp=r"C:\Users\13963\Documents\DLT\CNN"  # Filepath for input file : User Must Specify
ifp=path  # Filepath for input file 
ifn=r"\Data\As_All.csv" # Input filename for training dataset: User Must Specify
txtcol=2 # text column zero index
targcol=4 # target column zero index
ofp=path+r'\Output' # output file path
if not os.path.isdir(ofp):
    os.makedirs(ofp)
ofn=r'\CNN_1D_LearningCurve_'+ver+'.csv' # output file name
ofp_p=ofp # output file path
#ofp_p=r'C:\Users\13963\Documents\AI_Paper\Drafts\Revised_Submission_1\Output' # output file path
ofn_p=r'CNN_1D_' # output file name

csize=500 # chunk size for incrementing the training data
vsize=500 # val dataset size 
tsize=2500 # test dataset size
batch_size=128 # batch size
epochs=10 # num epochs



ifpn=ifp+ifn  # input filepath and name
df = pd.read_csv(ifpn, header=0) # read original data as dataframe
df=df.sample(frac=1,replace=False,random_state=45) # reshuffle with fixed seed
print (df.head(3))

header = list(df.columns.values) # original data header
txtcolhd=header[txtcol] # header of text column in input file
targcolhd=header[targcol] # header of target column in input file

df_text = df[txtcolhd] # text vector from training dataframe
df_text=df_text.tolist() # text vector as list
txt = [x.decode('ascii', 'ignore') for x in df_text if isinstance(x,basestring)] # convert to ascii         
df_target = df[targcolhd].values # target values as vector 

# second, prepare text samples and their labels
print('Processing text dataset')

texts=txt
labels=df_target
labels_index={'not relevant': 0,'relevant':1}

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    if word in voc: # if term in vocab
        embedding_matrix[i]=w2vmodel.wv[word] # look up w2v representation
    else: # if word not present in w2v model vocab
        embedding_matrix[i]=w2vmodel.wv['computer'] # use dummy word representation 


data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# carve out val and test data


X_val=data[0:vsize,:]
y_val=labels[0:vsize]
y_val = np.array(y_val).astype(int);

X_test=data[vsize:vsize+tsize,:]
y_test=labels[vsize:vsize+tsize]
y_test = np.array(y_test).astype(int);


X_r=data[tsize+vsize:,:] # residual X
y_r=labels[tsize+vsize:] # residual Y

print('Shape of X_val:', X_val.shape)
print('Shape of X_test:', X_test.shape)


finres=[] # initialize final results list 
for i in range(1,int(X_r.shape[0]/csize)+1): # loop over chunks carved out of residual X available for training
    X_train = X_r[:i*csize,:]    
    y_train=y_r[:i*csize]
    y_train = np.array(y_train).astype(int);
    print('Shape of X_train:', X_train.shape)
    finres.append(cnn_1D_fit()) # call cnn_fit function and append results



op = map(None, finres)    # output list
print (op) 

header_pr = ["n_Training","RunTime-Hrs","Predicted F1-score", "Actual F1-score","Predicted Precision", "Actual Precision", "Predicted Recall","Actual Recall","Predicted ROC-AUC","Actual ROC-AUC","Predicted PrRec-AUC","Actual PrRec-AUC"]    

ofpn=ofp+ofn
with open(ofpn, "w") as output: # Output results          
    writer = csv.writer(output, lineterminator='\n')    
    writer.writerow(header_pr)        
    for row in op:
        tempRow = row
        writer.writerow(tempRow)