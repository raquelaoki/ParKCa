import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
import time
import matplotlib.pyplot as plt

#path = 'C://Users//raoki//Documents//GitHub//ParKCa'
path = 'C://Users//raque//Documents//GitHub//ParKCa'
sys.path.append(path+'//src')
import CEVAE as cevae
import numpy.random as npr
from os import listdir
from os.path import isfile, join
os.chdir(path)
import random
randseed = 123
print("random seed: ", randseed)
random.seed(randseed)
np.random.seed(randseed)

pd.set_option('display.max_columns', 500)

from scipy.stats import ttest_ind,ttest_rel

n_units = 5000
n_causes = 10000# 10% var
SIMULATIONS = 10 


start_time = time.time()
train_path = 'data_s//snp_simulated_0.txt'
y01_path = 'data_s//snp_simulated_y01.txt'
y01 = np.asmatrix(pd.read_pickle(y01_path))[:,0]
#for t in range(SIMULATIONS):
    #CHECK INDEXS, I THINK IM DOING WRONG ASSOCIATIONS
at0 = []
at1 = []
cate = []
y01_predicted =[]
y01_ = []  

for (y0, y1, y10, y01_pred, yte) in cevae.model(n_causes,train_path,y01):
    at0.append(y0)
    at1.append(y1)
    cate.append(y10)
    y01_predicted.append(y01_pred)
    y01_.append(yte)
    if len(at0)%200 == 0: 
        output = pd.DataFrame({'at0':at0,
                               'at1':at1,
                               'cate':cate})
        
        predictions = np.stack( y01_predicted, axis=0 ).reshape(len(y01_predicted),len(y01_predicted[0]))
        testing_set = np.stack( y01_, axis=0 ).reshape(len(y01_),len(y01_[0]))
        predictions = pd.DataFrame(np.transpose(predictions))
        testing_set = pd.DataFrame(np.transpose(testing_set))
        output.to_pickle('results//simulations//cevae_output.txt')
        predictions.to_pickle('results//simulations//cevae_pred.txt')
        testing_set.to_pickle('results//simulations//cevae_test.txt')
        

output = pd.DataFrame({'at0':at0,'at1':at1,'cate':cate})

predictions = np.stack( y01_predicted, axis=0 ).reshape(len(y01_predicted),len(y01_predicted[0]))
testing_set = np.stack( y01_, axis=0 ).reshape(len(y01_),len(y01_[0]))
predictions = pd.DataFrame(np.transpose(predictions))
testing_set = pd.DataFrame(np.transpose(testing_set))
 
output.to_pickle('results//simulations//cevae_output.txt')
predictions.to_pickle('results//simulations//cevae_pred.txt')
testing_set.to_pickle('results//simulations//cevae_test.txt')
print("--- %s minutes ---" % str((time.time() - start_time)/60))