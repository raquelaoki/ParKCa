import sys
import os
import pandas as pd
import numpy as np

path = 'C://Users//raoki//Documents//GitHub//ParKCa'
sys.path.append(path+'//src')
import datapreprocessing as dp
import train as models
import experiments as exp
from os import listdir
from os.path import isfile, join
os.chdir(path)

pd.set_option('display.max_columns', 500)

APPLICATION1 = True #driver genes APPLICATION1

testing = True

if APPLICATION1:
    flag_first = True
    k_list = [15,30,45]
    if testing: k_list = k_list[0:1]
    pathfiles = path+'\\data'
    listfiles = [f for f in listdir(pathfiles) if isfile(join(pathfiles, f))]
    if testing: listfiles = listfiles[0:4]
    #filename = "data\\tcga_train_gexpression_cgc_7k.txt" #_2
    for filename in listfiles:
        train, j, v, y01, abr, colnames = dp.data_prep('data\\'+filename)
        if train.shape[0]>150:
            for k in k_list:
                name = 'd_'+filename.split("_")[-1].split(".")[0]
                ce0, roc, gamma0 = models.deconfounder(train,colnames,y01,name,k)
                roc['y01'] = y01
                roc.to_csv('results\\roc_'+name+'_'+str(k)+'.txt', sep=';', index = False)

                if flag_first:
                    ce = ce0
                    gamma = gamma0
                    flag_first = False
                else:
                    ce =pd.merge(ce, ce0,  how='outer', left_on='genes', right_on = 'genes')
                    gamma = pd.concat([gamma,gamma0],axis=0)
