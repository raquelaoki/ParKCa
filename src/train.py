import pandas as pd
import numpy as np
import warnings
import os
# import eval as eval
from os import listdir
from os.path import isfile, join
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import calibration
from scipy import sparse, stats

from torch.utils.data import Dataset, DataLoader
import torch as T
import torch.nn as nn
from torch import optim
import torch.distributions
import torch.nn.functional as F
from torch.distributions import bernoulli, normal
from torch.utils.data import WeightedRandomSampler

import tensorflow.compat.v1 as tf
from tensorflow.keras import optimizers
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd  # conda install -c conda-forge tensorflow-probability


tf.disable_v2_behavior()
tf.enable_eager_execution()
warnings.simplefilter("ignore")
# sys.path.insert(0, '/content/')



# import datapreprocessing as dp
# import CEVAE as cevae
# DA
# Meta-leaners packages
# https://github.com/aldro61/pu-learning (clone)
# from puLearning.puAdapter import PUAdapter
# https://github.com/t-sakai-kure/pywsl
# from pywsl.pul import pu_mr #pumil_mr (pip install pywsl)
# from pywsl.utils.syndata import gen_twonorm_pumil
# from pywsl.utils.comcalc import bin_clf_err


def learners(LearnersList, X, y, TreatCols=None, colnamesX=None, id='', Z=None, colnamesZ=None,
             path_output=None, cevaeMax=500, binfeatures=None, confeatures=None):
    """
    input:
        path_output: where to save the files
        X, colnamesX: potential causes and their names
        Z, colnamesZ: confounders and their names (clinical)
        y: 01 outcome
        causes: name of the potential causes (snps)
    """
    if TreatCols is None:
        TreatCols = list(range(X.shape[1]))
    roc_table = pd.DataFrame(columns=['learners', 'fpr', 'tpr', 'auc'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=63)
    coef_table = pd.DataFrame(columns=['causes'])
    coef_table['causes'] = ['T'+str(i) for i in range(len(TreatCols))]
    # ToDO: TreatCols implement
    # these are the columns I want to evaluate

    if 'DA' in LearnersList:
        print('\n\nLearner: DA')
        from deconfounder import deconfounder_algorithm as DA
        k_list = [15]  # if exploring multiple latent sizes
        print('... There are ', len(k_list), ' versions of DA')
        b = 100
        for k in k_list:
            print('...... Version 1/', len(k_list))
            coln = 'DA_' + str(id) + str(k)
            # coefk_table = pd.DataFrame(columns=[causes])
            model_da = DA(X_train, X_test, y_train, y_test, 10)
            coef, coef_continuos, roc = model_da.fit()
            roc_table = roc_table.append(roc, ignore_index=True)

            coef_table[coln] = coef_continuos[0:len(colnamesX)]
        print('Done!')
    if 'BART' in LearnersList:
        print('\n\nLearner: BART')
        from bart import BART as BART
        # model = SklearnModel(n_trees=50, n_burn=50, n_chains=1, n_jobs=1)  # Use default parameters
        # model.fit(x_snps, y)  # Fit the model
        model_bart = BART(X_train, X_test, y_train, y_test)
        model_bart.fit()
        print('...... predictions')
        coef_table['BART'] = model_bart.cate(TreatCols)
        # predictions = model.predict(x_snps)  # [:,0:1000] Make predictions on the train set
        # print(predictions[0])
        print('Done!')
    if 'CEVAE' in LearnersList:
        print('\n\n Learner: CEVAE')
        from cevae import CEVAE as CEVAE
        print('Note: Treatments should be the first columns of X')
        if colnamesZ is not None:
            nclinical = len(colnamesZ)
        else:
            nclinical = 0

        if cevaeMax < len(range(nclinical, nclinical+len(TreatCols))):
            print('... Working with dataset partitions')
            X_train_cevae_c, X_test_cevae_c = X_train[:, 0:nclinical], X_test[:, 0:nclinical]
            X_train_cevae_s = X_train[:, nclinical:X_train.shape[1]]
            X_test_cevae_s = X_test[:, nclinical:X_train.shape[1]]
            low = nclinical
            up = 0
            cate = []
            count = 0
            while up < X_train_cevae_s.shape[1]:
                up = np.min([low + cevaeMax, X_train_cevae_s.shape[1]])
                print('Partition', count, 'low  - up', low, up, ' Progress:', up*100/X_train_cevae_s.shape[1])
                count += 1
                #if count == 1:
                low = 0
                up = nclinical
                print('Partition', count, 'low  - up', low, up)
                X_train_cevae, X_test_cevae = X_train_cevae_s.copy(), X_test_cevae_s.copy()
                X_train_cevae, X_test_cevae = X_train_cevae[:, low:up], X_test_cevae[:, low:up]
                treatments = range(len(colnamesZ), len(colnamesZ) + up - low)
                X_train_cevae = np.concatenate([X_train_cevae_c, X_train_cevae], 1)
                X_test_cevae = np.concatenate([X_test_cevae_c, X_test_cevae], 1)
                model_cevae = CEVAE(X_train_cevae, X_test_cevae, y_train, y_test, treatments, binfeats=treatments,
                                    contfeats=range(len(colnamesZ)))
                out = model_cevae.fit_all()
                cate.append(out)
                low = up
                np.save('cevae_cate_checkpoints_z', cate)
                print('SAVED!')
                #else:
                #    low = up
            cate = [item for sublist in cate for item in sublist]
        else:
            print('Not using Partitions', cevaeMax, len(range(nclinical, nclinical+len(TreatCols))))
            # treatments = range(nclinical, nclinical+len(TreatCols))
            model_cevae = cevae(X_train, X_test, y_train, y_test, TreatCols,
                                binfeats=binfeatures, contfeats=confeatures)
            cate = model_cevae.fit_all()
        coef_table['CEVAE'] = cate
        np.save('level1data_learnersout_cevae', coef_table)
        print('Done!')
    if 'noise' in LearnersList:
        print('\n\nAdding noise')
        coef_table['noise'] = np.random.normal(0, 1, len(colnamesX))
        print(coef_table.head())
        #np.save('level1data_learnersout', coef_table)
    np.save('level1data_learnersout', coef_table)
    return coef_table


def meta_learner(level1data, MetaLearnerList, target='y_out', prob=1, ensemble=False, print_out=False):
    """
    Run the meta-learner
    input:
    - level 1 data (coeficients): learners output
    output:
    - roc table on the level 1 data: how well it identifies new causes
    - predictions: causal or not causal according to meta-learner
    - y_full_prob: prob of being causal - RF only for now
    """
    roc_table = pd.DataFrame(columns=['metalearners', 'pr_test',
                                      're_test', 'auc_test', 'f1_test',
                                      'pr_full', 're_full', 'f1_full'])
    # split data trainint and testing
    y = level1data[target]
    X = level1data.drop([target], axis=1)
    y_train, y_test, X_train, X_test, c_train, c_test = train_test_split(y, X, level1data.index.values,
                                                                         test_size=0.33, random_state=32)
    causes_order = np.concatenate([c_train, c_test], 0)

    # starting ensemble
    if ensemble:
        e_full_pred = np.zeros(len(y))
        e_test_pred = np.zeros(len(y_test))
        e = 0

    # Some causes are unknown or labeled as 0
    # Used on simulated dataset, when prob!=1
    if prob < 1:
        y_train = [i if np.random.binomial(1, prob, 1)[0] == 1 else 0 for i in y_train]
    else:
        pass

    y_train = pd.Series(y_train)
    predictions = pd.DataFrame(columns=MetaLearnerList)

    for m in MetaLearnerList:
        roc, y_test_pred, y_full_pred, y_full_prob = classification_models(y_train, y_test, X_train, X_test, m,
                                                                           print_out)
        roc_table = roc_table.append(roc, ignore_index=True)
        predictions[m] = y_full_pred

        if ensemble:
            if m == 'adapter' or m == 'upu' or m == 'lr' or m == 'rf' or m == 'nn':
                e_full_pred += y_full_pred
                e_test_pred += y_test_pred
                e += 1
            e_full_pred = np.divide(e_full_pred, e)
            e_test_pred = np.divide(e_test_pred, e)
            e_full_pred = [1 if i > 0.5 else 0 for i in e_full_pred]
            e_test_pred = [1 if i > 0.5 else 0 for i in e_test_pred]
            if print_out:
                print('... Testing set F1-score ', f1_score(y_test, e_test_pred))
                print('... Precision and Recall ', precision_score(y_test, e_test_pred),
                      recall_score(y_test, e_test_pred))
                print('... Confusion Matrix \n', confusion_matrix(y_test, e_test_pred))

                print('... Full set F1-score ', f1_score(y_full, e_full_pred))
                print('... Precision and Recall ', precision_score(y_full, e_full_pred),
                      recall_score(y_full, e_full_pred))
                print('... Confusion Matrix \n', confusion_matrix(y_full, e_full_pred))

            roc = {'metalearners': 'Ensemble',
                   'pr_test': precision_score(y_test, e_test_pred),
                   're_test': recall_score(y_test, e_test_pred),
                   'auc_test': roc_auc_score(y_test, e_test_pred),
                   'f1_test': f1_score(y_test, e_test_pred),
                   'pr_full': precision_score(y_full, e_full_pred),
                   're_full': recall_score(y_full, e_full_pred),
                   'f1_full': f1_score(y_full, e_full_pred)}
            roc_table = roc_table.append(roc, ignore_index=True)
    predictions['causes'] = causes_order
    predictions.set_index('causes', inplace=True)
    return roc_table, predictions, y_full_prob


def classification_models(y_train, y_test, X_train, X_test, name_model, print_out):
    """
    Classifiers used on the meta-learner
    Input:
       y_train, y_test, X_train, X_test: dataset to train and test the model
    output:
        - roc: roc on the known causes
        - y_test_pred
        - y_full_pred (train + test)
        - y_prob_full: predictions on X_full
    """
    X_full = np.concatenate((X_train, X_test), axis=0)
    y_full = np.concatenate((y_train, y_test), axis=0)
    warnings.filterwarnings("ignore")
    y_full_prob = None
    skip = False

    if name_model == 'nn':
        print('Meta-learner: NN')
        y_test_pred, y_full_pred = nn_classifier(y_train, X_train, X_test, X_full)
        pr = precision_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        if np.isnan(pr) or pr == 0 or f1 < 0.01:
            while np.isnan(pr) or pr == 0 or f1 < 0.001:
                print('...... trying again')
                y_test_pred, y_full_pred = nn_classifier(y_train, X_train, X_test, X_full)
    elif name_model == 'adapter':
        print('Meta-learner: PU-adapter')
        try:
            estimator = SVC(C=100, kernel='rbf', gamma='scale', probability=True)  # C = 0.3
            model = PUAdapter(estimator, hold_out_ratio=0.1)
            y0 = np.array(y_train)
            y0[np.where(y0 == 0)[0]] = -1
            model.fit(X_train, y0)
            y_test_pred, y_full_pred = model.predict(X_test), model.predict(X_full)
        except NameError:
            print('Library Missing')
            print("Check: https://github.com/aldro61/pu-learning")
            skip = True

    elif name_model == 'upu':
        """
        pul: nnpu (Non-negative PU Learning), pu_skc(PU Set Kernel Classifier),
        pnu_mr:PNU classification and PNU-AUC optimization (the one tht works: also use negative data)
        nnpu is more complicated (neural nets, other methos seems to be easier)
        try https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/pu_skc/demo_pu_skc.py
        and https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/upu/demo_upu.py
        """
        print('Meta-learner: UPU')
        try:
            prior = .48  # change for the proportion of 1 and 0
            param_grid = {'prior': [prior],
                          'lam': np.logspace(-3, 3, 5),  # what are these values
                          'basis': ['lm']}
            # upu (Unbiased PU learning)
            # https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/upu/demo_upu.py
            model = GridSearchCV(estimator=pu_mr.PU_SL(),
                                 param_grid=param_grid, cv=3, n_jobs=-1)
            model.fit(X_train, y_train)
            y_test_pred, y_full_pred = model.predict(X_test), model.predict(X_full)
        except NameError:
            print('Library Missing')
            print("Check: https://github.com/t-sakai-kure/pywsl")
            skip = True
    elif name_model == 'lr':
        print('Meta-learner: LR')
        from sklearn.linear_model import LogisticRegression
        w1 = y_train.sum() / len(y_train)
        w0 = 1 - w1
        sample_weight = {0: w1, 1: w0}
        model = LogisticRegression(C=0.1, penalty='l2', class_weight=sample_weight)  #
        model.fit(X_train, y_train)
        y_test_pred, y_full_pred = model.predict(X_test), model.predict(X_full)

    elif name_model == 'rf':
        print('Meta-learner: RF')
        w = len(y_train) / y_train.sum()
        sample_weight = np.array([w if i == 1 else 1 for i in y_train])
        model = RandomForestClassifier(max_depth=7, random_state=1)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        y_test_pred, y_full_pred = model.predict(X_test), model.predict(X_full)
        y_full_prob = model.predict_proba(X_full)
    else:
        print('Meta-learner: Random')
        w1 = y_train.sum() / len(y_train)
        # p_full = p / (len(y) + len(y_))
        y_test_pred = np.random.binomial(n=1, p=w1, size=X_test.shape[0])
        y_full_pred = np.random.binomial(n=1, p=w1, size=X_full.shape[0])

    if skip:
        roc = {'metalearners': name_model,
               'pr_test': np.nan,
               're_test': np.nan,
               'auc_test': np.nan,
               'f1_test': np.nan,
               'pr_full': np.nan,
               're_full': np.nan,
               'f1_full': np.nan}
        return roc, None, None, y_full_prob

    y_test_pred = np.where(y_test_pred == -1, 0, y_test_pred)
    y_full_pred = np.where(y_full_pred == -1, 0, y_full_pred)

    if print_out:
        print('... Testing set F1-score ', f1_score(y_test, y_test_pred))
        print('... Precision and Recall ', precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred))
        print('... Confusion Matrix \n', confusion_matrix(y_test, y_test_pred))

        print('... Full set F1-score ', f1_score(y_full, y_full_pred))
        print('... Precision and Recall ', precision_score(y_full, y_full_pred), recall_score(y_full, y_full_pred))
        print('... Confusion Matrix \n', confusion_matrix(y_full, y_full_pred))

    roc = {'metalearners': name_model,
           'pr_test': precision_score(y_test, y_test_pred),
           're_test': recall_score(y_test, y_test_pred),
           'auc_test': roc_auc_score(y_test, y_test_pred),
           'f1_test': f1_score(y_test, y_test_pred),
           'pr_full': precision_score(y_full, y_full_pred),
           're_full': recall_score(y_full, y_full_pred),
           'f1_full': f1_score(y_full, y_full_pred)}
    warnings.filterwarnings("default")
    return roc, y_test_pred, y_full_pred, y_full_prob


def nn_classifier(y_train, X_train, X_test, X_full):
    """
    meta-learner: not updated
    """

    # https://docs.microsoft.com/en-us/archive/msdn-magazine/2019/october/test-run-neural-binary-classification-using-pytorch
    class Batcher:
        def __init__(self, num_items, batch_size, seed=0):
            self.indices = np.arange(num_items)
            self.num_items = num_items
            self.batch_size = batch_size
            self.rnd = np.random.RandomState(seed)
            self.rnd.shuffle(self.indices)
            self.ptr = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.ptr + self.batch_size > self.num_items:
                self.rnd.shuffle(self.indices)
                self.ptr = 0
                raise StopIteration  # exit calling for-loop
            else:
                result = self.indices[self.ptr:self.ptr + self.batch_size]
                self.ptr += self.batch_size
                return result

    # ------------------------------------------------------------
    def akkuracy(model, data_x):
        # data_x and data_y are numpy array-of-arrays matrices
        X = T.Tensor(data_x)
        oupt = model(X)  # a Tensor of floats
        oupt = oupt.detach().float()
        oupt = [1 if i > 0.5 else 0 for i in oupt]
        return oupt

    # ------------------------------------------------------------
    class Net(T.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.hid1 = T.nn.Linear(size, 16)
            self.oupt = T.nn.Linear(16, 1)
            T.nn.init.xavier_uniform_(self.hid1.weight)
            T.nn.init.zeros_(self.hid1.bias)
            T.nn.init.xavier_uniform_(self.oupt.weight)
            T.nn.init.zeros_(self.oupt.bias)

        def forward(self, x):
            z = T.tanh(self.hid1(x))
            z = T.sigmoid(self.oupt(z))  # necessary
            return z

    size = X_train.shape[1]

    net = Net()

    net = net.train()  # set training mode
    lrn_rate = 0.01
    bat_size = 500
    loss_func = T.nn.BCELoss()  # softmax() + binary CE
    optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
    max_epochs = 500
    n_items = 8000
    batcher = Batcher(n_items, bat_size)
    X_test = X_test.values
    X_train = X_train.values
    # print('Starting training')

    # Divide by class
    df_class_0 = pd.DataFrame(X_train[y_train == 0])
    df_class_1 = pd.DataFrame(X_train[y_train == 1])
    df_class_0_under = df_class_0.sample(4000)
    df_class_1_over = df_class_1.sample(4000, replace=True)
    X_train2 = pd.concat([df_class_0_under, df_class_1_over], axis=0)
    X_train2['y'] = np.repeat([0, 1], 4000)
    X_train2 = X_train2.sample(frac=1).reset_index(drop=True)
    y_train2 = X_train2['y']
    X_train2 = X_train2.drop(['y'], axis=1)
    X_train2 = X_train2.values
    y_train2 = y_train2.values

    for epoch in range(0, max_epochs):
        for curr_bat in batcher:
            X = T.Tensor(X_train2[curr_bat])
            Y = T.Tensor(y_train2[curr_bat])
            optimizer.zero_grad()
            oupt = net(X)
            oupt = oupt.reshape(oupt.shape[0])
            # print(oupt.shape, Y.shape)
            loss_obj = loss_func(oupt, Y)
            loss_obj.backward()
            optimizer.step()
    # print('Training complete \n')
    net = net.eval()  # set eval mode

    y_pred = akkuracy(net, X_test)
    yfull = akkuracy(net, X_full)
    return y_pred, yfull


def data_norm(data1):
    """
    normalized data x- mean/sd
    input: dataset to be normalized
    output: normalized dataset
    """
    data1o = np.zeros(data1.shape)
    data1o[:, -1] = data1.iloc[:, -1]

    for i in range(0, data1.shape[1] - 1):
        nonzero = []
        for j in range(data1.shape[0]):
            if data1.iloc[j, i] != 0:
                nonzero.append(data1.iloc[j, i])
        for j in range(data1.shape[0]):
            if data1.iloc[j, i] != 0:
                data1o[j, i] = (data1.iloc[j, i] - np.mean(nonzero)) / np.sqrt(np.var(nonzero))

    data1o = pd.DataFrame(data1o)
    data1o.index = data1.index
    data1o.columns = data1.columns
    return data1o

