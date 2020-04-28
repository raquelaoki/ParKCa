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
        listfiles = listfiles[15:16]
        k = k_list[0]
        filename = listfiles[0]
        train, j, v, y01, abr, colnames = dp.data_prep('data\\'+filename)
        name = filename.split('_')[-1].split('.')[0]
        rating = pd.DataFrame(train)
        rating.colnames = colnames
        rating['id'] = rating.index
        rating = pd.melt(train,id_vars='id',var_name='genes', value_name='values')
        rating_ = []
        for i in range(rating.shape[0]):
            rating_.append([rating.id[i],rating.genes[i],rating.values[i]])

        #filename = "data\\tcga_train_gexpression_cgc_7k.txt" #_2
    else:
        for k in k_list:

             coefk_table = pd.DataFrame(columns=['genes'])
             roc_table = pd.DataFrame(columns=['learners', 'fpr','tpr','auc'])
             for filename in listfiles:
                 train, j, v, y01, abr, colnames = dp.data_prep('data\\'+filename)
                 if train.shape[0]>150:
                    print(filename,': ' ,train.shape[0])
                    #change filename
                    name = filename.split('_')[-1].split('.')[0]
                    coef, roc, coln = models.deconfounder_PPCA_LR(train,colnames,y01,name,k,b)
                    #organize columns names
                    roc_table = roc_table.append(roc,ignore_index=True)
                    coefk_table[coln] = coef

             print('--------- DONE ---------')
             coefk_table['genes'] = colnames

             roc_table.to_pickle('results//roc_'+str(k)+'.txt')
             coefk_table.to_pickle('results//coef_'+str(k)+'.txt')
             exp.roc_plot('results//roc_'+str(k)+'.txt')
