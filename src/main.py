import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")

path = 'C://Users//raoki//Documents//GitHub//ParKCa'
sys.path.append(path+'//src')
import datapreprocessing as dp
import train as models
import experiments as exp
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
        k = k_list[0]
        listfiles = listfiles[15:16]
        filename = listfiles[0]
        print('testing\n')
        train, j, v, y01, abr, colnames = dp.data_prep('data\\'+filename)
        print(filename,': ' ,train.shape[0])
        name = filename.split('_')[-1].split('.')[0]
        #input train,k output w,z, x_gen
        pmf = models.fm_PMF()
        pmf.set_params({"num_feat": k, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 10, "num_batches": 100,
                    "batch_size": 1000})
        
        ratings = pd.DataFrame(train)
        ratings.columns = colnames
        ratings['patient'] = ratings.index
        ratings = pd.melt(ratings,id_vars=['patient'],var_name='genes', value_name='values')
        ratings_ = []
        for i in range(ratings.shape[0]):
            ratings_.append([ratings.patient[i],
                             ratings.genes[i],
                             ratings.values[i]])


        print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
        x_train, x_test = train_test_split(ratings, test_size=0.2)  # spilt_rating_dat(ratings)
        pmf.fit(x_train, x_test)
        
        
        
        
        #filename = "data\\tcga_train_gexpression_cgc_7k.txt" #_2
    else:
        for k in k_list: 
              
             coefk_table = pd.DataFrame(columns=['genes'])
             roc_table = pd.DataFrame(columns=['learners', 'fpr','tpr','auc'])
             count = 0 
             for filename in listfiles:
                 print('\n'+str(count)+' of '+ str(len(listfiles)))
                 train, j, v, y01, abr, colnames = dp.data_prep('data\\'+filename)
                 if train.shape[0]>150:  
                    print(filename,': ' ,train.shape[0])
                    #change filename
                    name = filename.split('_')[-1].split('.')[0]
                    coef, roc, coln = models.deconfounder_PPCA_LR(train,colnames,y01,name,k,b)
                    #organize columns names
                    roc_table = roc_table.append(roc,ignore_index=True)
                    coefk_table[coln] = coef
                 else:
                    print(filename,'(SKIP): ' ,train.shape[0])
            
             print('--------- DONE ---------')
             coefk_table['genes'] = colnames
             
             roc_table.to_pickle('results//roc_'+str(k)+'.txt')
             coefk_table.to_pickle('results//coef_'+str(k)+'.txt')
             exp.roc_plot('results//roc_'+str(k)+'.txt')                  
                    
