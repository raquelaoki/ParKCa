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
testing = True

if APPLICATION:
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
        #coef, roc, coln = models.deconfounder_PPCA_LR(train,colnames,y01,name,k,b)
        coef, roc, coln = models.BART(train,colnames, y01,name,False,'results\\bart_model.sav')


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
                    load = False 
                    coef, roc, coln = models.deconfounder_PPCA_LR(train,colnames,y01,name,load)
                    #organize columns names
                    roc_table = roc_table.append(roc,ignore_index=True)
                    coefk_table[coln] = coef

             print('--------- DONE ---------')
             coefk_table['genes'] = colnames

             roc_table.to_pickle('results//roc_'+str(k)+'.txt')
             coefk_table.to_pickle('results//coef_'+str(k)+'.txt')
             exp.roc_plot('results//roc_'+str(k)+'.txt')

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
    #True causes and lambda, now i need to add the noise? 
    import numpy.random as npr
    tc_ = npr.normal(loc = 0 , scale=1, size=n_causes)
    #True causes 
    tc = [i if i>np.quantile(tc_,0.99) else 0 for i in b_]#truncate so only 1% are different from 0 
    sigma = np.zeros(n_units)
    sigma = [4*4 if lambdas[j]==0 else sigma[j] for j in range(len(sigma))]
    sigma = [7*7 if lambdas[j]==1 else sigma[j] for j in range(len(sigma))]
    sigma = [2*2 if lambdas[j]==2 else sigma[j] for j in range(len(sigma))]
    y0 = np.array(tc).reshape(1,-1).dot(np.transpose(G))
    y1 = 30*lambdas.reshape(1,-1)
    y2 = npr.normal(0,sigma,n_units).reshape(1,-1)
    y = y0 + y1 + y2
    p = 1/(1+np.exp(y0 + y1 + y2))
    print(np.var(y),np.var(y0),np.var(y1),np.var(y2))
    print('This should be 10%: ', np.var(y0)/np.var(y-y0))
    print('This should be 20%: ', np.var(y1)/np.var(y-y1))
    print('This should be 70%: ', np.var(y2)/np.var(y-y2))
    #del y0, y1, y2,y
    y01 = np.zeros(len(p[0])) 
    y01 = [npr.binomial(1,p[0][i],1)[0] for i in range(len(p[0]))]
    y01 = np.asarray(y01)
    #568 1's
    print(sum(y01), len(y01))
    #train, j, v, y01, abr, colnames
    train_s = np.asmatrix(G)
    j, v = G.shape
    abr = []

    #make a fundtion 
    colnames = []
    causes = 0 
    noncauses = 0 
    for i in range(len(tc)):
        if tc[i]>0: 
            colnames.append('causal_'+str(causes))
            causes+=1
        else: 
            colnames.append('noncausal_'+str(noncauses))
            noncauses+=1
    
    name = 'simulation1'
    print(name,': ' ,train_s.shape[0])
                    #change filename
    k = 15
    b = 10 
    G = pd.DataFrame(G)
    G.columns = colnames
    del colnames
    #end function 
    
    
    #application still works 
    #train_s needs to by a nympy matrix
    #train[3:5,3:5] Out[9]: 
    #matrix([[1.60943791, 6.63200178],
    #        [1.38629436, 6.52795792]])
    #type(colnames) = pandas.core.indexes.base.Index
    #Index(['A1CF'], dtype='object')
    #type(y01) = numpy.ndarray
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
    