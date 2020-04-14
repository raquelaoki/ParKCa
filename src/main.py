import sys
import os
import pandas as pd
import numpy as np

path = 'C://Users//raoki//Documents//GitHub//ParKCa'
sys.path.append(path+'//src')
import datapreprocessing as dp
import train as models
import experiments as exp
os.chdir(path)

pd.set_option('display.max_columns', 500)

APPLICATION1 = True #driver genes APPLICATION1

if APPLICATION1:
    filename = "data\\tcga_train_gexpression_cgc_7k.txt" #_2
    train, j, v, y01, abr, colnames = dp.data_prep(filename)

    k = 40
    a1, a2, a3 = models.deconfounder(train,colnames,y01,'d_all',k)
