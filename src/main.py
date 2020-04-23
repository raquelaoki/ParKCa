import sys
import os
import pandas as pd
import numpy as np

path = 'C://Users//raoki//Documents//GitHub//ParKCa'
sys.path.append(path+'//src')
import datapreprocessing as dp
import train as models
#import experiments as exp
from os import listdir
from os.path import isfile, join
os.chdir(path)
import random
randseed = 123
print("random seed: ", randseed)
random.seed(randseed)
np.random.seed(randseed)

pd.set_option('display.max_columns', 500)

APPLICATION1 = True #driver genes APPLICATION1

testing = True

if APPLICATION1:
    flag_first = True
    k_list = [15,30,45]
    if testing: k_list = k_list[0:1]
    pathfiles = path+'\\data'
    listfiles = [f for f in listdir(pathfiles) if isfile(join(pathfiles, f))]
    if testing:
         filename = listfiles[15]
         roc_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
         train, j, v, y01, abr, colnames = dp.data_prep('data\\'+filename)
         #add dataset name 
         coef, roc = models.deconfounder_PPCA_LR(train,colnames,y01,'da_ppca_lr',20,10)
         roc_table.append(roc,ignore_index=True)
         #x_train, x_val, holdout_mask,holdout_row = models.daHoldout(train,0.2)
         #w,z, x_gen = models.fm_PPCA(x_train,10)
         #pvalue= models.daPredCheck(x_val,x_gen,w,z, holdout_mask,holdout_row)
         #print(pvalue)

    else:
    #filename = "data\\tcga_train_gexpression_cgc_7k.txt" #_2
        b =100 
        for k in k_list: 
             coefk_table = []
             roc_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

             for filename in listfiles:
                 train, j, v, y01, abr, colnames = dp.data_prep('data\\'+filename)
                 if train.shape[0]>150:  
                    #change filename
                    coef, roc = models.deconfounder_PPCA_LR(train,colnames,y01,'da_ppca_lr',k,b)
                    #organize columns names
                    roc_table.append(roc,ignore_index=True)
                    coefk_table.append(coef)
             np.savetxt('results//coef_'+k+'.txt', coefk_table, delimiter=',') 
             np.savetxt('results//roc_'+k+'.txt', roc_table, delimiter=',') 
                    
                    
                    
                    
