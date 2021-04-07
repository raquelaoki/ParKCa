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
from CompBioAndSimulated_Datasets.simulated_data_multicause import *
sys.path.insert(0, 'ParKCa/src/')
from ParKCa.src.train import *

randseed = 123
print("random seed: ", randseed)
random.seed(randseed)
np.random.seed(randseed)
warnings.simplefilter("ignore")
pd.set_option('display.max_columns', 500)


# TODO: update with main format and config.yaml
# todo: tcga
# todo: fix proportion of known causes

def main():
    """
    Real-world application
    level 0 data: gene expression of patients with cancer
    level 0 outcome: metastasis
    """
    results_level1, results_final = [],[]
    with open(config_path) as f:
        config = yaml.load_all(f, Loader=yaml.FullLoader)
        for p in config:
            params = p["parameters"]

    if 'tcga' in params['data']:
        print("\n\n\n STARTING EXPERIMENTS ON APPLICATION")
        X, y01 = read('TCGA.txt') #not tested
        # BART is tested using an R code
        level1data = learners(['DA', 'BART'], pd.DataFrame(X), y01, TreatCols=treatement_columns,
                          colnamesX=colnamesX)


        cgc_list = dp.cgc('cancer_gene_census.csv')
        level1data['y'] = [1 if i in cgc_list else 0 for i in level1data['gene']]
        level1data.set_index('gene', inplace=True, drop=True)

        level1data = dp.data_norm(level1data)

        # DIVERSITY
        # fix column names
        qav, q_ = eval.diversity(['bart_all', 'bart_FEMALE', 'bart_MALE'],
                                 ['dappcalr_15_LGG', 'dappcalr_15_SKCM', 'dappcalr_15_all', 'dappcalr_15_FEMALE',
                                  'dappcalr_15_MALE'],
                                 level1data)
        print('DIVERSITY: ', qav)

        # Metalearners
        experiments1 = models.meta_learner(level1data, ['adapter', 'upu', 'lr', 'rf', 'nn', 'random'], 1)
        experiments0 = eval.first_level_asmeta(['bart_all', 'bart_FEMALE', 'bart_MALE'],
                                               ['dappcalr_15_LGG', 'dappcalr_15_SKCM', 'dappcalr_15_all',
                                                'dappcalr_15_FEMALE', 'dappcalr_15_MALE'],
                                               data1)

        #experiments1.to_csv('results\\eval_metalevel1.txt', sep=';')
        #experiments1c.to_csv('results\\eval_metalevel1c.txt', sep=';')
        experiments0.to_csv('results\\eval_metalevel0.txt', sep=';')
        print("DONE WITH EXPERIMENTS ON APPLICATION")
        results_level1.append(level1data)
        results_final.append(experiments1)

    elif 'gwas' in params['data']:
        sdata_gwas = gwas_simulated_data(prop_tc=0.05,
                                         pca_path='/content/CompBioAndSimulated_Datasets/data/tgp_pca2.txt')
        X, y, y01, treatement_columns, treatment_effects, group = sdata_gwas.generate_samples()

        colnamesX = ['Col' + str(i) for i in range(X.shape[1])]
        output = learners(['CEVAE', 'DA', 'BART'], pd.DataFrame(X), y01, TreatCols=treatement_columns,
                          colnamesX=colnamesX)
        output['true'] = treatment_effects[treatement_columns]

        known_snps = [] #not tested
        level1data['y'] = 0
        level1data['y'] = [1 if i in known_snps else 0 for i in level1data['causes'].values]
        level1data.set_index('causes', inplace=True, drop=True)
        level1data = data_norm(level1data)
        roc, output, y_full_prob = meta_learner(level1data.fillna(0), ['lr', 'nn', 'upu', 'rf'], 'y')
        results_level1.append(output)
        results_final.append(output)

    return results_level1, results_final

if __name__ == '__main__':
    # main(config_path = sys.argv[1])
    main(config_path='/content/ParKCa')
