import sys
import os
import pandas as pd
import numpy as np
import warnings
import time
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import CEVAE as cevae
import bart as bart
import deconfounder as decondouder
import train as models
import eval as eval
import numpy.random as npr
from os import listdir
from os.path import isfile, join
from scipy.stats import ttest_ind, ttest_rel

path = 'C://Users//raque//Documents//GitHub//ParKCa'
sys.path.append(path + '//extra')
os.chdir(path)
randseed = 123
print("random seed: ", randseed)
random.seed(randseed)
np.random.seed(randseed)
warnings.simplefilter("ignore")
pd.set_option('display.max_columns', 500)

#TODO: update with main format and config.yaml


if EVALUATION_A:
    '''
    Real-world application
    level 0 data: gene expression of patients with cancer
    level 0 outcome: metastasis
    '''
    print("\n\n\n STARTING EXPERIMENTS ON APPLICATION")

    #BART is tested using an R code
    models.learners(APPLICATIONBOOL=True,DABOOL=True, BARTBOOL=False, CEVAEBOOL=False,path = path)

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
    experiments1 = models.meta_learner(data1, ['adapter','upu','lr','rf','nn','random'],1)
    experiments1c = models.meta_learner(data1c, ['adapter','upu','lr','rf','nn','random'],1)

    experiments0 = eval.first_level_asmeta(['bart_all',  'bart_FEMALE',  'bart_MALE' ],
                        ['dappcalr_15_LGG','dappcalr_15_SKCM','dappcalr_15_all','dappcalr_15_FEMALE','dappcalr_15_MALE'],
                        data1)


    experiments1.to_csv('results\\eval_metalevel1.txt', sep=';')
    experiments1c.to_csv('results\\eval_metalevel1c.txt', sep=';')
    experiments0.to_csv('results\\eval_metalevel0.txt', sep=';')
    print("DONE WITH EXPERIMENTS ON APPLICATION")


if __name__ == '__main__':
    # main(config_path = sys.argv[1])
    main(config_path='/content/')