import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
EVALUATION_A = False
EVALUATION_S = True
#cuda test torch.cuda.FloatTensor(2)

'''
Real-world application
level 0 data: gene expression of patients with cancer
level 0 outcome: metastasis
'''
#models.learners(APPLICATIONBOOL=True,DABOOL=True, BARTBOOL=False, CEVAEBOOL=False,path = path)

if EVALUATION_A:
    features_bart =  pd.read_csv("results\\coef_bart.txt",sep=';')
    features_da15 = pd.read_pickle("results\\coef_15.txt")
    features_da15c = pd.read_pickle("results\\coefcont_15.txt")

    level1data = features_bart.merge(features_da15,  left_on='gene', right_on='genes').drop(['genes'],1)
    level1datac = features_bart.merge(features_da15c,  left_on='gene', right_on='genes').drop(['genes'],1)

    cgc_list = dp.cgc('extra\\cancer_gene_census.csv')
    level1data = cgc_list.merge(level1data, left_on='genes',right_on='gene',how='right').drop(['genes'],1)
    level1datac = cgc_list.merge(level1datac, left_on='genes',right_on='gene',how='right').drop(['genes'],1)

    level1data['y_out'].fillna(0,inplace=True)
    level1datac['y_out'].fillna(0,inplace=True)

    level1data.set_index('gene', inplace = True, drop = True)
    level1datac.set_index('gene', inplace = True, drop = True)


    data1 = dp.data_norm(level1data)
    data1c = dp.data_norm(level1datac)

    #DIVERSITY
    qav, q_ = eval.diversity(['bart_all',  'bart_FEMALE',  'bart_MALE' ],
                        ['dappcalr_15_LGG','dappcalr_15_SKCM','dappcalr_15_all','dappcalr_15_FEMALE','dappcalr_15_MALE'],
                        data1)
    print('DIVERSITY: ', qav)
    #Metalearners
    experiments1 = models.meta_learner(data1, ['adapter','upu','lr','rf','random'],1)
    experiments1c = models.meta_learner(data1c, ['adapter','upu','lr','rf','random'],1)

    experiments0 = eval.first_level_asmeta(['bart_all',  'bart_FEMALE',  'bart_MALE' ],
                        ['dappcalr_15_LGG','dappcalr_15_SKCM','dappcalr_15_all','dappcalr_15_FEMALE','dappcalr_15_MALE'],
                        data1)
    #models.meta_learner(data1, ['lr'])


    experiments1.to_csv('results\\eval_metalevel1.txt', sep=';')
    experiments1c.to_csv('results\\eval_metalevel1c.txt', sep=';')
    experiments0.to_csv('results\\eval_metalevel0.txt', sep=';')

    #more layers and nodes improved
    from sklearn.model_selection import train_test_split,  GridSearchCV, StratifiedKFold
    
    
    count_class_0, count_class_1 = data1.y_out.value_counts()
    
    # Divide by class
    df_class_0 = data1[data1['y_out'] == 0]
    df_class_1 = data1[data1['y_out'] == 1]
    
    df_class_0_under = df_class_0.sample(4000)
    df_class_1_over = df_class_1.sample(4000, replace=True)
    data2 = pd.concat([df_class_0_under, df_class_1_over], axis=0)
    
    
    y = data1['y_out']
    X = data1.drop(['y_out'], axis=1)
    
    
    y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.33,random_state=22)
    models.nn_classifier(y_train, y_test, X_train, X_test, 1000,10,0.001)


'''
Simulation
level 0 data: Binary treatments
level 0 outcome: Binary
'''

if EVALUATION_S: 
    
    #SAVING 10 datasets
    n_units = 5000
    n_causes = 10000# 10% var
    sim = 10
    
    #dp.generate_samples(sim,n_units, n_causes)
    
    
    #DA
    pathtc = 'data_s\\snp_simulated1_truecauses.txt'
    pathy01 = 'data_s\\snp_simulated1_y01.txt'
    tc  = pd.read_pickle(pathtc)
    y01 = pd.read_pickle(pathy01)
    
    
    #CREATE THE DATASET
    #dp.sim_level1data([0,1,2,3,4,5,6,7,8,9],tc,y01,'sim_roc_simulations')
    #eval.roc_plot('results//sim_roc_simulations.txt')
    eval.simulation_eval(10)
    


#ROC SCORE FOR CEVAE 


#file1 = 'results//simulations//pred_y_for_roc//cevae_pred'
#file2 = 'results//simulations//pred_y_for_roc//cevae_test'
#eval.roc_cevae(file1,file2,10,'cevae')

