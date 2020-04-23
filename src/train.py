import pandas as pd
import numpy as np
import warnings

from os import listdir
from os.path import isfile, join
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix,f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import roc_curve
from sklearn import linear_model


from scipy.stats import gamma
from scipy import sparse, stats
import statsmodels.discrete.discrete_model as sm

import tensorflow as tf #.compat.v2
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import functools



def deconfounder_PPCA_LR(train,colnames,y01,name,k,b):
    '''
    input:
    - train dataset
    - colnames or possible causes
    - y01: outcome
    - name: file name
    - k: dimension of latent space
    - b: number of bootstrap samples

    '''
    x_train, x_val, holdout_mask,holdout_row = models.daHoldout(train,0.2)

    pca,z, x_gen = models.fm_PPCA(x_train,k)
    filename = name+'_' +str(k)
    pvalue= models.daPredCheck(x_val,x_gen,pca,z, holdout_mask,holdout_row)
    if 0.1 < pvalue and pvalue < 0.9:
        print('Pass Predictive Check')
        for i in range(b):
            rows = np.random.choice(train.shape[0], int(train.shape[0]*0.85), replace=False)
            X = train[rows, :]
            y01_b = y01[rows]
            w,pca, x_gen = models.fm_PPCA(X,k)
            #outcome model
            result , pred, extra = outcome_model_ridge(X,colnames, Z,y01_b,False)
        #df_ce =pd.merge(df_ce, causal_effect,  how='left', left_on='genes', right_on = 'genes')
        #df_roc[name_PCA]=roc
        #aux = pd.DataFrame({'model':[name_PCA],'gamma':[gamma],'gamma_l':[cil],'gamma_u':[cip]})
        #df_gamma = pd.concat([df_gamma,aux],axis=0)
        #df_gamma[name_PCA] = sparse.coo_matrix((gamma_ic),shape=(1,3)).toarray().tolist()

    return pvalue

def fm_MF(train,k):
    '''
    Matrix Factorization to extract latent features
    Parameters:
        train: dataset
        k: latent Dimension
    Return:
        2 matrices
    '''
    model = NMF(n_components=k, init='random') #random_state=0
    W = model.fit_transform(train)
    H = model.components_

    return W, H

def fm_PPCA(train,latent_dim):
    num_datapoints, data_dim = train.shape
    x_train = tf.convert_to_tensor(np.transpose(train),dtype = tf.float32)


    Root = tfd.JointDistributionCoroutine.Root
    def probabilistic_pca(data_dim, latent_dim, num_datapoints, stddv_datapoints):
      w = yield Root(tfd.Independent(
          tfd.Normal(loc=tf.zeros([data_dim, latent_dim]),
                     scale=2.0 * tf.ones([data_dim, latent_dim]),
                     name="w"), reinterpreted_batch_ndims=2))
      z = yield Root(tfd.Independent(
          tfd.Normal(loc=tf.zeros([latent_dim, num_datapoints]),
                     scale=tf.ones([latent_dim, num_datapoints]),
                     name="z"), reinterpreted_batch_ndims=2))
      x = yield tfd.Independent(tfd.Normal(
          loc=tf.matmul(w, z),
          scale=stddv_datapoints,
          name="x"), reinterpreted_batch_ndims=2)

    #data_dim, num_datapoints = x_train.shape
    stddv_datapoints = 1

    concrete_ppca_model = functools.partial(probabilistic_pca,
        data_dim=data_dim,
        latent_dim=latent_dim,
        num_datapoints=num_datapoints,
        stddv_datapoints=stddv_datapoints)

    model = tfd.JointDistributionCoroutine(concrete_ppca_model)

    #actual_w, actual_z, x_train = model.sample()


    w = tf.Variable(np.ones([data_dim, latent_dim]), dtype=tf.float32)
    z = tf.Variable(np.ones([latent_dim, num_datapoints]), dtype=tf.float32)

    target_log_prob_fn = lambda w, z: model.log_prob((w, z, x_train))
    losses = tfp.math.minimize(lambda: -target_log_prob_fn(w, z),
                               optimizer=tf.optimizers.Adam(learning_rate=0.05),
                               num_steps=200)

    qw_mean = tf.Variable(np.ones([data_dim, latent_dim]), dtype=tf.float32)
    qz_mean = tf.Variable(np.ones([latent_dim, num_datapoints]), dtype=tf.float32)
    qw_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([data_dim, latent_dim]), dtype=tf.float32))
    qz_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([latent_dim, num_datapoints]), dtype=tf.float32))
    def factored_normal_variational_model():
      qw = yield Root(tfd.Independent(tfd.Normal(
          loc=qw_mean, scale=qw_stddv, name="qw"), reinterpreted_batch_ndims=2))
      qz = yield Root(tfd.Independent(tfd.Normal(
          loc=qz_mean, scale=qz_stddv, name="qz"), reinterpreted_batch_ndims=2))

    surrogate_posterior = tfd.JointDistributionCoroutine(
        factored_normal_variational_model)

    losses = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn,
        surrogate_posterior=surrogate_posterior,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        num_steps=400)


    x_generated = []
    for i in range(50):
        _, _, x_g = model.sample(value=surrogate_posterior.sample(1))
        x_generated.append(x_g.numpy()[0])

    w, z = surrogate_posterior.variables

    return w.numpy(),z.numpy(), x_generated

def daHoldout(train,holdout_portion):
    num_datapoints, data_dim = train.shape
    n_holdout = int(holdout_portion * num_datapoints * data_dim)

    holdout_row = np.random.randint(num_datapoints, size=n_holdout)
    holdout_col = np.random.randint(data_dim, size=n_holdout)
    holdout_mask = (sparse.coo_matrix((np.ones(n_holdout), \
                                (holdout_row, holdout_col)), \
                                shape = train.shape)).toarray()

    holdout_subjects = np.unique(holdout_row)
    holdout_mask = np.minimum(1, holdout_mask)

    x_train = np.multiply(1-holdout_mask, train)
    x_vad = np.multiply(holdout_mask, train)
    return x_train, x_vad,holdout_mask,holdout_row

def daPredCheck(x_val,x_gen,w,z,holdout_mask,holdout_row):
    #obs_ll = []
    #rep_ll = []
    x = np.multiply(np.transpose(np.dot(w,z)), holdout_mask)
    holdout_subjects = np.unique(holdout_row)
    holdout_mask1 = np.asarray(holdout_mask).reshape(-1)
    #x_val1 = np.squeeze(np.asarray(x_val))
    x_val1 = np.asarray(x_val).reshape(-1)
    #x_val1 = np.reshape(x_val, [1,-1])[0]
    x1 = np.asarray(x).reshape(-1)
    x_val1 = x_val1[holdout_mask1==1]
    x1= x1[holdout_mask1==1]
    pvals =[]

    for i in range(len(x_gen)):
        generate = np.transpose(x_gen[i])
        holdout_sample = np.multiply(generate, holdout_mask)
        holdout_sample = np.asarray(holdout_sample).reshape(-1)
        holdout_sample = holdout_sample[holdout_mask1==1]
        x_val_current = stats.norm(holdout_sample, 1).logpdf(x_val1)
        x_gen_current = stats.norm(holdout_sample, 1).logpdf(x1)

        #print(stats.norm(holdout_sample, 1).logpdf(x_val)[0,3],x_val[0,3],stats.norm(holdout_sample, 1).logpdf(x)[0,3],x[0,3],' g:',holdout_sample[0,3])
        #print(stats.norm(holdout_sample, 1).logpdf(x_val)[0,0],x_val[0,0],stats.norm(holdout_sample, 1).logpdf(x)[0,0],x[0,0],' g:',holdout_sample[0,2])

        #obs_ll.append(x_val_current)
        #rep_ll.append(x_gen_current)
        pvals.append(np.mean(np.array(x_val_current<x_gen_current)))


    overall_pval = np.mean(pvals)
    return overall_pval

def check_save(Z,train,colnames,y01,name1,name2,k):
    '''
    Run predictive check function and print results
    input:
        z: latent features
        train: training set
        colnames: genes names
        y01: binary classification
        name: name for the file
        k: size of the latent features (repetitive)
    output:
        save features on results folder or print that it failed
        return the predicted values for the training data on the outcome model
    '''
    colname1 = name2+'_'+name1+'_'+str(k)
    gamma,cil,cip, test_result = predictive_check(train,Z)
    #result = []
    #roc = []
    if(test_result):
        print('Predictive Check test: PASS',colname1)
        #result , pred = outcome_model( train,colnames, Z,y01,colname1)
        result , pred, extra = outcome_model_ridge(train,colnames, Z,y01,colname1)
        if(len(pred)!=0):
            #resul.to_csv('results\\feature_'+name1+'_'+str(k)+'_lr_'+name2+'.txt', sep=';', index = False)
            #name = name1+str(k)+'_lr_'+name2
            roc = pred
        else:
            roc = []
            print('Outcome Model Does Not Coverge, results are not saved')
            np.savetxt('results\\FAIL_outcome_feature_'+name1+'_'+str(k)+'_lr_'+name2+'.txt',[], fmt='%s')

    else:
        print('Predictive Check Test: FAIL',colname1)
        np.savetxt('results\\FAIL_pcheck_feature_'+name1+'_'+str(k)+'_lr_'+name2+'.txt',[], fmt='%s')
    return result, roc, gamma,cil,cip#, name1+'_'+str(k)+'_lr_'+name2

def outcome_model_ridge(x, colnames,x_latent,y01_b,roc_flag,name):
    '''
    input:
    - x: training set
    - x_latent: output from factor model
    - colnames: x colnames or possible causes
    - y01: outcome
    -name: roc name file
    '''
    import scipy.stats as st
    x_aug = np.concatenate([x,np.transpose(x_latent)],axis=1)
    ridge = linear_model.RidgeClassifierCV(scoring='roc_auc',cv =5, normalize = True)
    ridge.fit(x_aug, y01_b)
    coef = ridge.coef_[0][0:X.shape[1]]
    
    if roc_flag: 
        pred = ridge.decision_function(x_aug)
        fpr, tpr, _ = roc_curve(y01_b, pred)
        auc = roc_auc_score(y01_b, pred)
        roc = {'classifiers':cls.name,
               'fpr':fpr, 
               'tpr':tpr, 
               'auc':auc}

    else:
        roc = {}
    #resul = pd.DataFrame({'genes':colnames,colname1+'_pvalue': coef_pvalues,colname1+'_coef':coef_mean })

    #extra = pd.DataFrame({'genes':colnames,colname1+'_icl': coef_low,colname1+'_icu':coef_up })

    return coef, roc
