import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")

#path = 'C://Users//raoki//Documents//GitHub//ParKCa'
path = 'C://Users//raque//Documents//GitHub//ParKCa'

sys.path.append(path+'//src')
import datapreprocessing as dp
import train as models
import experiments as exp
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

APPLICATION = True #driver genes APPLICATION1
SIMULATION = False
testing = False
DA = True
BART = False

if APPLICATION:
    k_list = [15,30]
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
        #coef, roc, coln = models.deconfounder_PPCA_LR(train,colnames,y01,name,k,b)
        coef, roc, coln = models.BART(train,colnames, y01,name,False)


    else:
        if DA:
            print('DA')
            skip = ['CHOL','LUSC','HNSC','PRAD'] #F1 score very low
            for k in k_list:
                 coefk_table = pd.DataFrame(columns=['genes'])
                 roc_table = pd.DataFrame(columns=['learners', 'fpr','tpr','auc'])
                 #test
                 for filename in listfiles:
                     train, j, v, y01, abr, colnames = dp.data_prep('data\\'+filename)
                     if train.shape[0]>150:
                        print(filename,': ' ,train.shape[0])
                        #change filename
                        name = filename.split('_')[-1].split('.')[0]
                        if name not in skip:
                            coef, roc, coln = models.deconfounder_PPCA_LR(train,colnames,y01,name,k,b)
                            roc_table = roc_table.append(roc,ignore_index=True)
                            coefk_table[coln] = coef

                 print('--------- DONE ---------')
                 coefk_table['genes'] = colnames

                 roc_table.to_pickle('results//roc_'+str(k)+'.txt')
                 coefk_table.to_pickle('results//coef_'+str(k)+'.txt')
                 exp.roc_plot('results//roc_'+str(k)+'.txt')

        if BART:
            print('BART')
            coefk_table = pd.DataFrame(columns=['genes'])
            roc_table = pd.DataFrame(columns=['learners', 'fpr','tpr','auc'])
            for filename in listfiles:
                train, j, v, y01, abr, colnames = dp.data_prep('data\\'+filename)
                if train.shape[0]>150:
                   print(filename,': ' ,train.shape[0])
                   name = filename.split('_')[-1].split('.')[0]
                   load = True
                   coef, roc, coln = models.BART(train,colnames, y01,name,load)
                   roc_table = roc_table.append(roc,ignore_index=True)
                   coefk_table[coln] = coef
            print('--------- DONE ---------')
            coefk_table['genes'] = colnames

            roc_table.to_pickle('results//roc_'+'bart'+'.txt')
            coefk_table.to_pickle('results//coef_'+'bart'+'.txt')
            exp.roc_plot('results//roc_'+'bart'+'.txt')


if SIMULATION:
    #ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/
    vcf_path = "data_s//ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.vcf.gz"
    h5_path = 'data_s//ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.h5'
    #sim_load_vcf_to_h5(vcf_path,h5_path)
    #S = dp.sim_load_h5_to_PCA(h5_path)
    S = np.loadtxt('data_s//tgp_pca2.txt', delimiter=',')
    n_units = 5000
    n_causes = 10000# 10% var

    G, lambdas = dp.sim_genes_TGP([], [], 0 , n_causes, n_units, S, D=3)
    G = sim_dataset(G,lambdas, n_causes)
    G = dp.add_colnames(G,tc)

    train_s = np.asmatrix(G)
    j, v = G.shape
    name = 'simulation1'
    print(name,': ' ,train_s.shape[0])
                    #change filename
    k = 15
    b = 10
    coef, roc, coln = models.deconfounder_PPCA_LR(train_s,G.columns,y01,name,k,10)
    #roc_table = roc_table.append(roc,ignore_index=True)
    #coefk_table[coln] = coef
    tc01 =[1 if tc[i]>0 else 0 for i in range(len(tc))]
    coef01 = [1 if coef[i]>0 else 0 for i in range(len(coef))]
    from sklearn.metrics import confusion_matrix,f1_score, accuracy_score
    confusion_matrix(tc01,coef01)
    #Results are bad, but I think there is hope
    #Copy others from application
    #exp.roc_plot('results//sroc_'+str(k)+'.txt')
