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
    k_list = [15,30]
    if testing: k_list = k_list[0:1]
    pathfiles = path+'\\data'
    listfiles = [f for f in listdir(pathfiles) if isfile(join(pathfiles, f))]
    b =100
    if testing:
        b = int(b/10)
        listfiles = listfiles[0:6]
        k_list = k_list[0:1]
        
    #filename = "data\\tcga_train_gexpression_cgc_7k.txt" #_2
         
    for k in k_list: 
        
         coefk_table = pd.DataFrame(columns=['genes'])
         roc_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

         for filename in listfiles:
             train, j, v, y01, abr, colnames = dp.data_prep('data\\'+filename)
             if train.shape[0]>150:  
                #change filename
                name = filename.split('_')[-1].split('.')[0]
                coef, roc, coln = models.deconfounder_PPCA_LR(train,colnames,y01,name,k,b)
                #organize columns names
                roc_table = roc_table.append(roc,ignore_index=True)
                coefk_table[coln] = coef
        
         coefk_table['genes'] = colnames
        
         np.savetxt('results//coef_'+str(k)+'.txt', coefk_table, delimiter=',') 
         np.savetxt('results//roc_'+str(k)+'.txt', roc_table, delimiter=',') 
                    
                    
                    
                    
