import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")

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
from sklearn.metrics import roc_curve,roc_auc_score
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
    x_train, x_val, holdout_mask = daHoldout(train,0.2)
    w,z, x_gen = fm_PPCA(x_train,k,True)
    filename = 'dappcalr_' +str(k)+'_'+name
    pvalue= daPredCheck(x_val,x_gen,w,z, holdout_mask)
    alpha = 0.05 #for the IC test on outcome model 
    low = stats.norm(0,1).ppf(alpha/2)
    up = stats.norm(0,1).ppf(1-alpha/2)
    #To speed up, I wont fit the PPCA to each boostrap iteration
    del x_gen
    if 0.1 < pvalue and pvalue < 0.9:
        print('Pass Predictive Check:', filename )
        coef= []
        pca = np.transpose(z)
        for i in range(b):
            #print(i)
            rows = np.random.choice(train.shape[0], int(train.shape[0]*0.85), replace=False)
            X = train[rows, :]
            y01_b = y01[rows]
            pca_b = pca[rows,:] 
            #w,pca, x_gen = fm_PPCA(X,k)
            #outcome model
            coef_, _ = outcome_model_ridge(X,colnames, pca_b,y01_b,False,filename)
            coef.append(coef_)
        
        
        coef = np.matrix(coef)
        #Building IC 
        coef_m = np.asarray(np.mean(coef,axis=0)).reshape(-1)
        coef_var = np.asarray(np.var(coef,axis=0)).reshape(-1)
        coef_z = np.divide(coef_m,np.sqrt(coef_var/b))
        #1 if significative and 0 otherwise
        coef_z = [ 1 if c>low and c<up else 0 for c in coef_z ] 
        
        
        #https://abdalimran.github.io/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot
        #Calculating ROC with entire train    
        del X,pca,pca_b,y01_b
        del coef_var, coef, coef_
        w,z, x_gen = fm_PPCA(train,k,False)    
        _,roc =  outcome_model_ridge(train,colnames, np.transpose(z),y01,True,filename)
        #df_ce =pd.merge(df_ce, causal_effect,  how='left', left_on='genes', right_on = 'genes')
        #df_roc[name_PCA]=roc
        #aux = pd.DataFrame({'model':[name_PCA],'gamma':[gamma],'gamma_l':[cil],'gamma_u':[cip]})
        #df_gamma = pd.concat([df_gamma,aux],axis=0)
        #df_gamma[name_PCA] = sparse.coo_matrix((gamma_ic),shape=(1,3)).toarray().tolist()
    else: 
        coef_m = []
        coef_z = []
        roc = []

    return np.multiply(coef_m,coef_z), roc, filename

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

def fm_PPCA(train,latent_dim, flag_pred):
    #source: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_PCA.ipynb
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
    if flag_pred:
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
    return x_train, x_vad,holdout_mask

def daPredCheck(x_val,x_gen,w,z,holdout_mask):
    #obs_ll = []
    #rep_ll = []
    #holdout_subjects = np.unique(holdout_row)
    holdout_mask1 = np.asarray(holdout_mask).reshape(-1)
    x_val1 = np.asarray(x_val).reshape(-1)
    x1 = np.asarray(np.multiply(np.transpose(np.dot(w,z)), holdout_mask)).reshape(-1)
    del x_val
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

        pvals.append(np.mean(np.array(x_val_current<x_gen_current)))


    overall_pval = np.mean(pvals)
    return overall_pval


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
    ridge = linear_model.RidgeClassifierCV(scoring='roc_auc',cv =5, normalize = True)
    
    if roc_flag: 
        #use testing and training set
        x_aug = np.concatenate([x,x_latent],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(x_aug, y01_b, test_size=0.33, random_state=42) 
       
        ridge.fit(X_train, y_train)
        coef = ridge.coef_[0][0:x.shape[1]]
    

        print(f1_score(y_test,ridge.predict(X_test)))
        pred = ridge.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, pred)
        auc = roc_auc_score(y01_b, pred)
        roc = {'learners': name,
               'fpr':fpr, 
               'tpr':tpr, 
               'auc':auc}

    else:
        #don't split dataset 
        x_aug = np.concatenate([x,x_latent],axis=1)
        ridge.fit(x_aug, y01_b)
        coef = ridge.coef_[0][0:x.shape[1]]
        roc = {}
    #resul = pd.DataFrame({'genes':colnames,colname1+'_pvalue': coef_pvalues,colname1+'_coef':coef_mean })

    #extra = pd.DataFrame({'genes':colnames,colname1+'_icl': coef_low,colname1+'_icu':coef_up })

    return coef, roc
