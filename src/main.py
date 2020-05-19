import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
import time
import matplotlib.pyplot as plt


path = 'C://Users//raoki//Documents//GitHub//ParKCa'
#path = 'C://Users//raque//Documents//GitHub//ParKCa'

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
#models.learners(APPLICATIONBOOL=True,DABOOL=True, BARTBOOL=True, CEVAEBOOL=False,path)

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


'''
if CEVAE:
    #roc_table = pd.DataFrame(columns=['learners', 'fpr','tpr','auc'])
    #12:16, 14/05/2020
    #testing
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
        if len(at0)%100 == 0:
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


    #y01_pred_ = [1 if i >0.5 else 0 for i in y_pred]
    #f1_score(yte,y01_pred_)
    #f1_score(yte,npr.binomial(1,yte.sum()/len(yte),size=len(yte)))

    ##coef, roc, coln = models.deconfounder_PPCA_LR(train_s,G.columns,y01,name,k,10)
    #roc_table = roc_table.append(roc,ignore_index=True)
    #coefk_table[coln] = coef
    ##tc01 =[1 if tc[i]>0 else 0 for i in range(len(tc))]
    ##coef01 = [1 if coef[i]>0 else 0 for i in range(len(coef))]
    ##from sklearn.metrics import confusion_matrix,f1_score, accuracy_score
    ##confusion_matrix(tc01,coef01)
    #Results are bad, but I think there is hope
    #Copy others from application
    #exp.roc_plot('results//sroc_'+str(k)+'.txt')
'''
