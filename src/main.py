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
#import datapreprocessing as dp
import CEVAE as cevae
#import train as models
#import eval as eval
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

APPLICATION = False #driver genes APPLICATION1
SIMULATION = False
testing = True
DA = False
BART = False
CEVAE = True


#SAVING 10 datasets 
n_units = 5000
n_causes = 10000# 10% var
SIMULATIONS = 10 
if SIMULATION:
    #ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/
    vcf_path = "data_s//ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.vcf.gz"
    h5_path = 'data_s//ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.h5'
    #sim_load_vcf_to_h5(vcf_path,h5_path)
    #S = dp.sim_load_h5_to_PCA(h5_path)
    S = np.loadtxt('data_s//tgp_pca2.txt', delimiter=',')

    
    sim_y = []
    sim_tc = []
    for sim in range(SIMULATIONS):    
        G0, lambdas = dp.sim_genes_TGP([], [], 0 , n_causes, n_units, S, 3, sim )
        G1, tc, y01 = dp.sim_dataset(G0,lambdas, n_causes,n_units,sim)
        G = dp.add_colnames(G1,tc)
        del G0,G1
    
        #train_s = np.asmatrix(G)
        #j, v = G.shape
        #print(name,': ' ,train_s.shape[0])
        G.to_pickle('data_s//snp_simulated_'+str(sim)+'.txt')
        sim_y.append(y01)
        sim_tc.append(tc)
    sim_y = np.transpose(np.matrix(sim_y))
    sim_y = pd.DataFrame(sim_y)
    sim_y.columns = ['sim_'+str(sim) for sim in range(SIMULATIONS)]
    
    sim_tc = np.transpose(np.matrix(sim_tc))
    sim_tc = pd.DataFrame(sim_tc)
    sim_tc.columns = ['sim_'+str(sim) for sim in range(SIMULATIONS)]
    
    sim_y.to_pickle('data_s//snp_simulated_y01.txt')
    sim_tc.to_pickle('data_s//snp_simulated_truecauses.txt')

         #change filename
    #k = 15
    #b = 10
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
