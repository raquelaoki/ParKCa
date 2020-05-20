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
sys.path.append(path+'//extra')
import datapreprocessing as dp
#import CEVAE as cevae
import train as models
import eval as eval
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

SIMULATION = False
testing = False
DA = False
BART = True
CEVAE = False


'''
Real-world application
level 0 data: gene expression of patients with cancer
level 0 outcome: metastasis
'''
#models.learners(APPLICATIONBOOL=True,DABOOL=True, BARTBOOL=False, CEVAEBOOL=False,path = path)

features_bart =  pd.read_csv("results\\coef_bart.txt",sep=';')
features_da15 = pd.read_pickle("results\\coef_15.txt")
level1data = features_bart.merge(features_da15,  left_on='gene', right_on='genes').drop(['genes'],1)
cgc_list = dp.cgc('extra\\cancer_gene_census.csv')
level1data = cgc_list.merge(level1data, left_on='genes',right_on='gene',how='right').drop(['genes'],1)
level1data['y_out'].fillna(0,inplace=True)
level1data.set_index('gene', inplace = True, drop = True)


data1 = dp.data_norm(level1data)
data1.head()

#DIVERSITY

#Metalearners
experiments1 = models.meta_learner(data1, ['adapter','upu','lr','rf','random'])
experiments2 = eval.first_level_asmeta(['bart_all',  'bart_FEMALE',  'bart_MALE' ],
                        ['dappcalr_15_LGG','dappcalr_15_SKCM','dappcalr_15_all','dappcalr_15_FEMALE','dappcalr_15_MALE'],
                        data1)


experiments1.to_csv('results\\eval_metalevel1.txt', sep=';')
experiments2.to_csv('results\\eval_metalevel0.txt', sep=';')

#more layers and nodes improved
#models.nn_classifier(y_train, y_test, X_train, X_test, 100,64,0.001)


'''
Simulation
level 0 data: Binary treatments
level 0 outcome: Binary
'''


#SAVING 10 datasets
n_units = 5000
n_causes = 10000# 10% var
sim = 10

#dp.generate_samples(sim,n_units, n_causes)


#CEVAE CODE ON NOTEBOOK - CHANGE NAME
#sim1_cevae_out = dp.join_simulation('results\\simulations\\cevae_output_sim1_',['a','b','c'])
#sim1 = join_simulation('results\\simulations\\cevae_output_sim1_',['a','b','c'])

#DA
pathtc = 'data_s\\snp_simulated1_truecauses.txt'
pathy01 = 'data_s\\snp_simulated1_y01.txt'
tc  = pd.read_pickle(pathtc)
y01 = pd.read_pickle(pathy01)


for i in range(9):
    sim = 'sim_'+str(i)
    tc_sim1 = tc[sim]
    tc_sim1_bin = [1 if i != 0 else 0 for i in tc_sim1]
    y01_sim1 = y01[sim]

    train = pd.read_pickle('data_s\\snp_simulated1_'+str(i)+'.txt')
    coef, roc, coln = models.deconfounder_PPCA_LR(np.asmatrix(train),train.columns,y01_sim1,sim,15,100)
    
    #Join CEVAE results
    cavae = dp.join_simulation(path = 'results\\simulations\\cevae_output_sim'+str(i)+'_', files = ['a','b'])
    
    
    data = pd.DataFrame({'cevae':cavae['cate'],'coef':coef,'y_out':tc_sim1_bin})
    models.meta_learner(data,['rf','lr','random','upu'])
    eval.first_level_asmeta(['cevae' ],['coef'],data)
    

