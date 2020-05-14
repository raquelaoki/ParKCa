import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
import eval as eval
import datapreprocessing as dp
#import CEVAE as cevae

from os import listdir
from os.path import isfile, join

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix,f1_score, accuracy_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn import linear_model
from sklearn import calibration

from scipy.stats import gamma
from scipy import sparse, stats

import statsmodels.discrete.discrete_model as sm

#DA
from tensorflow.keras import optimizers
import tensorflow as tf #.compat.v2
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
tf.enable_eager_execution()
import functools

#Learners
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
        print('Pass Predictive Check:', filename, '(',str(pvalue),')' )
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
        coef = coef[:,0:train.shape[1]]
        #Building IC
        coef_m = np.asarray(np.mean(coef,axis=0)).reshape(-1)
        coef_var = np.asarray(np.var(coef,axis=0)).reshape(-1)
        coef_z = np.divide(coef_m,np.sqrt(coef_var/b))
        #1 if significative and 0 otherwise
        coef_z = [ 1 if c>low and c<up else 0 for c in coef_z ]


        #https://abdalimran.github.io/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot
        '''
        if ROC = TRUE, outcome model receive entire dataset, but internally split in training
        and testing set. The ROC results and score is just for testing set
        '''
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

def fm_PPCA(train,latent_dim, flag_pred):
    #Reference: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_PCA.ipynb
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
                               optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
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
    model = linear_model.SGDClassifier(penalty='l2', alpha=0.1, l1_ratio=0.01,loss='modified_huber', fit_intercept=True,random_state=0)


    #ridge = linear_model.RidgeClassifierCV(scoring='roc_auc',cv =5, normalize = True)

    if roc_flag:
        #use testing and training set
        x_aug = np.concatenate([x,x_latent],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(x_aug, y01_b, test_size=0.33, random_state=42)
        modelcv = calibration.CalibratedClassifierCV(base_estimator=model, cv=5, method='isotonic').fit(X_train, y_train)
        coef = []

        pred = modelcv.predict(X_test)
        predp = modelcv.predict_proba(X_test)
        predp1 = [i[1] for i in predp]
        print('F1:',f1_score(y_test,pred),sum(pred),sum(y_test))
        fpr, tpr, _ = roc_curve(y_test, predp1)
        auc = roc_auc_score(y_test, predp1)
        roc = {'learners': name,
               'fpr':fpr,
               'tpr':tpr,
               'auc':auc}


    else:
        #don't split dataset
        x_aug = np.concatenate([x,x_latent],axis=1)
        model.fit(x_aug, y01_b)
        coef = model.coef_[0]
        roc = {}
    #resul = pd.DataFrame({'genes':colnames,colname1+'_pvalue': coef_pvalues,colname1+'_coef':coef_mean })

    #extra = pd.DataFrame({'genes':colnames,colname1+'_icl': coef_low,colname1+'_icu':coef_up })

    return coef, roc

def learners(APPLICATIONBOOL, DABOOL, BARTBOOL, CEVAEBOOL,path ):
    '''
    Function to run the application.
    INPUT:
    bool variables
    path: path for level 0 data
    OUTPUT:
    plots, coefficients and roc data (to_pickle format)

    NOTE: This code does not run the BART, it only reads the results.
    BART model was run using R
    '''
    if APPLICATION:
        k_list = [15,30]
        pathfiles = path+'\\data'
        listfiles = [f for f in listdir(pathfiles) if isfile(join(pathfiles, f))]
        b =100

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
                 eval.roc_plot('results//roc_'+str(k)+'.txt')

        if BART:
            print('BART')
            #MODEL AND PREDICTIONS MADE ON R
            filenames=['results//bart_all.txt','results//bart_MALE.txt','results//bart_FEMALE.txt']
            eval.roc_table_creation(filenames,'bart')
            eval.roc_plot('results//roc_'+'bart'+'.txt')

        if BART and DA
        filenames=['results//roc_bart.txt','results//roc_15.txt']
        eval.roc_plot_all(filenames)

#Meta-learner
def class_models(y,y_test,X,X_test,name_model):
    """
    Input:
        X,y,X_test, y_test: dataset to train the model
    Return:
        cm: confusion matrix for the testing set
        cm_: confusion matrix for the full dataset
        y_all_: prediction for the full dataset
    """
    X_full = np.concatenate((X,X_test), axis = 0 )
    y_full = np.concatenate((y,y_test), axis = 0 )
    warnings.filterwarnings("ignore")
    if name_model == 'nn':
        #IMPLEMENT

    elif name_model == 'adapter':
        #keep prob
        #csv don't have nu
        print('adapter',X.shape[1])
        #it was c=0.5
        estimator = SVC(C=0.3, kernel='rbf',gamma='scale',probability=True)
        model = PUAdapter(estimator, hold_out_ratio=0.3)
        X = np.matrix(X)
        y = np.array(y)
        model.fit(X, y)

    elif name_model == 'upu':
        '''
        pul: nnpu (Non-negative PU Learning), pu_skc(PU Set Kernel Classifier),
        pnu_mr:PNU classification and PNU-AUC optimization (the one tht works: also use negative data)
        nnpu is more complicated (neural nets, other methos seems to be easier)
        try https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/pu_skc/demo_pu_skc.py
        and https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/upu/demo_upu.py
         '''
        print('upu', X.shape[1])
        #Implement these, packages only work on base anaconda (as autoenconder)
        #https://github.com/t-sakai-kure/pywsl
        prior =.5 #change for the proportion of 1 and 0
        param_grid = {'prior': [prior],
                          'lam': np.logspace(-3, 1, 5), #what are these values
                          'basis': ['lm']}
        lambda_list = np.logspace(-3, 1, 5)
        #upu (Unbiased PU learning)
        #https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/upu/demo_upu.py
        model = GridSearchCV(estimator=pu_mr.PU_SL(),
                               param_grid=param_grid, cv=10, n_jobs=-1)
        X = np.matrix(X)
        y = np.array(y)
        model.fit(X, y)

    elif name_model == 'lr':
        print('lr',X.shape[1])
        model = sm.Logit(y,X).fit_regularized(method='l1')

    elif name_model=='rf':
        print('rd',X.shape[1])
        md = max(np.floor(X.shape[1]/3),6)
        model = RandomForestClassifier(max_depth=md, random_state=0)
        model.fit(X, y)

    else:
        print('random',X.shape[1])

    if name_model=='random':
        y_ = np.random.binomial(n=1,p=y.sum()/len(y),size =X_test.shape[0])
        y_full_ = np.random.binomial(n=1,p=y.sum()/len(y),size=X_full.shape[0])
    else:
        y_ = model.predict(X_test)
        y_full_ = model.predict(X_full)

    if name_model == 'lr':
        #y_ = 1- y_
        #y_full_ = 1- y_full_
        y_[y_<0.5] = 0
        y_[y_>=0.5] = 1
        y_full_[y_full_< 0.5] = 0
        y_full_[y_full_>=0.5] = 1

    y_ = np.where(y_==-1,0,y_)
    y_full_ = np.where(y_full_==-1, 0,y_full_)

    acc = accuracy_score(y_test,y_)
    acc_f = accuracy_score(y_full, y_full_)
    f1 = f1_score(y_test,y_)
    f1_f = f1_score(y_full, y_full_)
    tnfpfntp = confusion_matrix(y_test,y_).ravel()
    tnfpfntp_= confusion_matrix(y_full, y_full_).ravel()
    tp_genes = np.multiply(y_full, y_full_)
    warnings.filterwarnings("default")
    return [acc, acc_f, f1, f1_f], tnfpfntp, tnfpfntp_, tp_genes,y_,y_full_

def data_running_models(data_list, names, name_in, name_out, is_bin, id):
    '''
    input: list with combinations of features and the names of the datsets
    outout:
    '''
    acc_ , acc = [] , []
    f1_, f1 = [],[]
    tnfpfntp,tnfpfntp_ = [],[] #confusion_matrix().ravel()
    tp_genes = []
    model_name, data_name = [],[]
    nin, nout = [],[]
    error = []
    size = []
    id_name = []
    models = ['adapter','upu','lr','rf','nn','random']
    for dt,dtn in zip(data_list,names):
        if dt.shape[1]>2:
            #print('type: ',dt,dtn,'shape:', dt.shape[1], dt.head())
            #dt['y_out'].fillna(0,inplace = True)
            y = dt['y_out'].fillna(0)
            X = dt.drop(['y_out'], axis=1)
            index_save = X.index
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            X = pd.DataFrame(X,index=index_save)
            y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.3)
            index_ = [list(X_train.index),list(X_test.index)]
            flat_index = [item for sublist in index_ for item in sublist]
            flat_index = np.array(flat_index)
            #print('INDEX',len(flat_index),len(list(X_train.index)),len(list(X_test.index)))
            e_full_ = np.where(y==1,0,0)
            e_ = np.where(y_test==1,0,0)
            ensemble_c = 0
            for m in models:
                try:
                    scores, cm, cm_, tp_genes01, y_,y_full_ = pul(y_train, y_test, X_train, X_test,'name',m)
                    acc.append(scores[0])
                    acc_.append(scores[1])
                    f1.append(scores[2])
                    f1_.append(scores[3])
                    tnfpfntp.append(cm)
                    tnfpfntp_.append(cm_)
                    tp_genes.append(flat_index[np.equal(tp_genes01,1)])
                    model_name.append(m)
                    data_name.append(dtn)
                    nin.append(name_in)
                    nout.append(name_out)
                    error.append(False)
                    if(m=='adapter' or m=='upu' or m=='lr' or m=='randomforest'):
                        e_full_ = e_full_+y_full_
                        e_ = e_+y_
                        ensemble_c = ensemble_c+1
                except:
                    acc.append(np.nan)
                    acc_.append(np.nan)
                    f1.append(np.nan)
                    f1_.append(np.nan)
                    tnfpfntp.append([np.nan,np.nan,np.nan,np.nan])
                    tnfpfntp_.append([np.nan,np.nan,np.nan,np.nan])
                    tp_genes.append([])
                    model_name.append(m)
                    data_name.append(dtn)
                    nin.append(name_in)
                    nout.append(name_out)
                    error.append(True)
                    print('Error in PUL model',m,dtn)
                size.append(X_train.shape[1])
                id_name.append(id)

            #print('test',ensemble_c,acc)
            e_full_ = np.multiply(e_full_,1/ensemble_c)
            e_ = np.multiply(e_,1/ensemble_c)
            e_full_ = np.where(np.array(e_full_)>0.5,1,0)
            e_ = np.where(np.array(e_)>0.5,1,0)
            y_full = np.concatenate((y_train,y_test), axis = 0 )
            acc.append(accuracy_score(y_test,e_))
            acc_.append(accuracy_score(y_full,e_full_))
            f1.append(f1_score(y_test,e_))
            f1_.append(f1_score(y_full,e_full_))
            tnfpfntp.append(confusion_matrix(y_test,e_).ravel())
            tnfpfntp_.append(confusion_matrix(y_full,e_full_).ravel())
            tp_genes.append([])
            model_name.append('ensemble')
            data_name.append(dtn)
            nin.append(name_in)
            nout.append(name_out)
            error.append(False)
            size.append(X_train.shape[1])
            id_name.append(id)
        else:
            print(dtn, 'only one columns')
    dt_exp = pd.DataFrame({'acc':acc,'acc_':acc_, 'f1':f1, 'f1_':f1_,
                               'tnfpfntp':tnfpfntp, 'tnfpfntp_':tnfpfntp_,
                               'tp_genes':tp_genes,'model_name':model_name , 'data_name':data_name,
                               'nin':nin, 'nout':nout, 'error':error, 'size': size,
                               'id':id_name})
    return dt_exp
