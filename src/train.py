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
import functools

#Meta-leaners packages
#https://github.com/aldro61/pu-learning
from puLearning.puAdapter import PUAdapter
#https://github.com/t-sakai-kure/pywsl
from pywsl.pul import pumil_mr, pu_mr
from pywsl.utils.syndata import gen_twonorm_pumil
from pywsl.utils.comcalc import bin_clf_err

import statsmodels.discrete.discrete_model as sm
from sklearn.ensemble import RandomForestClassifier
#NN
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
    
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
        #np.multiply(coef_m,coef_z)
    return coef_m, roc, filename

def fm_PPCA(train,latent_dim, flag_pred):
    #Reference: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_PCA.ipynb
    from tensorflow.keras import optimizers
    import tensorflow as tf #.compat.v2
    import tensorflow_probability as tfp
    from tensorflow_probability import distributions as tfd
    tf.enable_eager_execution()

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
        print('Confusion Matrix', confusion_matrix(y_test,pred))
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
    if APPLICATIONBOOL:
        k_list = [15,30]
        pathfiles = path+'\\data'
        listfiles = [f for f in listdir(pathfiles) if isfile(join(pathfiles, f))]
        b =100

        if DABOOL:
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
                            coef, roc, coln = deconfounder_PPCA_LR(train,colnames,y01,name,k,b)
                            roc_table = roc_table.append(roc,ignore_index=True)
                            coefk_table[coln] = coef
                        else:
                            print('skip',name)

                 print('--------- DONE ---------')
                 coefk_table['genes'] = colnames

                 #CHANGE HERE 20/05 
                 #roc_table.to_pickle('results//roc_'+str(k)+'.txt')
                 coefk_table.to_pickle('results//coef2_'+str(k)+'.txt')
                 #eval.roc_plot('results//roc_'+str(k)+'.txt')

        if BARTBOOL:
            print('BART')
            #MODEL AND PREDICTIONS MADE ON R
            filenames=['results//bart_all.txt','results//bart_MALE.txt','results//bart_FEMALE.txt']
            eval.roc_table_creation(filenames,'bart')
            eval.roc_plot('results//roc_'+'bart'+'.txt')

        if BARTBOOL and DABOOL:
            #filenames=['results//roc_bart.txt','results//roc_15.txt']
            eval.roc_plot_all(filenames)

#Meta-learner
def classification_models(y,y_,X,X_,name_model):
    """
    Input:
        X,y,X_test, y_test: dataset to train the model
    Return:
        cm: confusion matrix for the testing set
        cm_: confusion matrix for the full dataset
        y_all_: prediction for the full dataset
    """

    warnings.filterwarnings("ignore")
    if name_model == 'nn':
        #IMPLEMENT
        print('a')
    elif name_model == 'adapter':
        #keep prob
        #it was c=0.5
        estimator = SVC(C=0.3, kernel='rbf',gamma='scale',probability=True)
        model = PUAdapter(estimator, hold_out_ratio=0.1)
        X = np.matrix(X)
        y0 = np.array(y)
        y0[np.where(y0 == 0)[0]] = -1
        model.fit(X, y0)

    elif name_model == 'upu':
        '''
        pul: nnpu (Non-negative PU Learning), pu_skc(PU Set Kernel Classifier),
        pnu_mr:PNU classification and PNU-AUC optimization (the one tht works: also use negative data)
        nnpu is more complicated (neural nets, other methos seems to be easier)
        try https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/pu_skc/demo_pu_skc.py
        and https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/upu/demo_upu.py
         '''
        print('upu', X.shape[1])
        #https://github.com/t-sakai-kure/pywsl
        prior =.5 #change for the proportion of 1 and 0
        param_grid = {'prior': [prior],
                          'lam': np.logspace(-3, 3, 5), #what are these values
                          'basis': ['lm']}
        #upu (Unbiased PU learning)
        #https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/upu/demo_upu.py
        model = GridSearchCV(estimator=pu_mr.PU_SL(),
                               param_grid=param_grid, cv=3, n_jobs=-1)
        X = np.matrix(X)
        y = np.array(y)
        model.fit(X, y)

    elif name_model == 'lr':
        print('lr',X.shape[1])
        X = np.matrix(X)
        y = np.array(y)
        #model = sm.Logit(y,X).fit_regularized(method='l1')
        from sklearn.linear_model import LogisticRegression
        w1 = y.sum()/len(y)
        w0 = 1 - w1
        sample_weight = {0:w1,1:w0}
        model = LogisticRegression(C=.1,class_weight=sample_weight,penalty='l2') #
        model.fit(X,y)
        
        #p = LogisticRegression(C=1e9,class_weight=sample_weight).fit(X_train,y_train).predict(X_train)


    

    elif name_model=='rf':
        print('rd',X.shape[1])
        #md = max(np.floor(X.shape[1]/3),6)
        w = len(y)/y.sum()
        sample_weight = np.array([w if i == 1 else 1 for i in y])
        model = RandomForestClassifier(max_depth=12, random_state=0)
        model.fit(X, y,sample_weight = sample_weight)
        #solver='lbfgs'

    else:
        print('random',X.shape[1])


    X_full = np.concatenate((X,X_), axis = 0 )
    y_full = np.concatenate((y,y_), axis = 0)

    if name_model=='random':
         p = y.sum()+y_.sum()
         p_full = p/(len(y)+len(y_))
         y_pred = np.random.binomial(n=1,p=y_.sum()/len(y_),size =X_.shape[0])
         ypred = np.random.binomial(n=1,p=p_full,size =X_full.shape[0])
    else:
        y_pred = model.predict(X_)
        ypred = model.predict(X_full)

    if name_model =='uajfiaoispu':
        print(y_pred)
        print('\nTesting set: \n',confusion_matrix(y_,y_pred))
        print('\nFull set: \n',confusion_matrix(y_full,ypred))
        print('\nPrecision ',precision(1,confusion_matrix(y_,y_pred)))
        print('Recall',recall(1,confusion_matrix(y_,y_pred)))

    #if name_model == 'lr':
     #   print(y_pred.sum(),y_pred)
     #   y_pred = [0 if i<0.5 else 1 for i in y_pred]
     #   ypred = [0 if i<0.5 else 1 for i in ypred]

    #Some models pred -1 instead of 0
    y_pred = np.where(y_pred==-1,0,y_pred)
    ypred = np.where(ypred==-1,0,ypred)

    #fpr, tpr, _ = roc_curve(y_,y_pred)
    pr = precision(1,confusion_matrix(y_,y_pred))
    re = recall(1,confusion_matrix(y_,y_pred))
    auc = roc_auc_score(y_,y_pred)
    f1 = f1_score(y_full,ypred)
    f1_ = f1_score(y_,y_pred)

    #tp_genes = np.multiply(y_full, y_full_)
    roc = {'metalearners': name_model,'precision':pr ,'recall':re,'auc':auc,'f1':f1,'f1_':f1_}
    warnings.filterwarnings("default")
    return roc, ypred, y_pred

def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()

def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def meta_learner(data1, models):
    '''
    input: level 1 data
    outout:
    '''
    roc_table = pd.DataFrame(columns=['metalearners', 'precision','recall','auc','f1','f1_'])
    tp_genes = []

    #split data trainint and testing
    y = data1['y_out']
    X = data1.drop(['y_out'], axis=1)
    y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.33,random_state=22)

    #starting ensemble
    e_full = np.zeros(len(y))
    e_pred = np.zeros(len(y_test))
    e = 0

    for m in models:
        roc, yfull, y_pred = classification_models(y_train, y_test, X_train, X_test,m)
        #tp_genes.append(flat_index[np.equal(tp_genes01,1)])
        roc_table = roc_table.append(roc,ignore_index=True)
        #ensemble
        if(m=='adapter' or m=='upu' or m=='lr' or m=='rf' or m=='nn'):
            e_full += yfull
            e_pred += y_pred
            e += 1

    #finishing ensemble
    e_full = np.divide(e_full,e)
    e_pred = np.divide(e_pred,e)
    e_full= [1 if i>0.5 else 0 for i in e_full]
    e_pred= [1 if i>0.5 else 0 for i in e_pred]

    #fpr, tpr, _ = roc_curve(y_test,e_pred)
    pr = precision(1,confusion_matrix(y_test,e_pred))
    re = recall(1,confusion_matrix(y_test,e_pred))
    auc = roc_auc_score(y_test,e_pred)
    f1 = f1_score(np.hstack([y_test,y_train]),e_full)
    f1_ = f1_score(y_test,e_pred)
    roc = {'metalearners': 'ensemble','precision':pr ,'recall':re,'auc':auc,'f1':f1,'f1_':f1_}
    roc_table = roc_table.append(roc,ignore_index=True)
    return roc_table


def nn_classifier(y_train, y_test, X_train, X_test, EPOCHS, BATCH_SIZE, LEARNING_RATE):
    
    
    #https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89

    
    class trainData(Dataset):
        
        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data
            
        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]
            
        def __len__ (self):
            return len(self.X_data)
    
    
    train_data = trainData(torch.FloatTensor(X_train.to_numpy()), 
                           torch.FloatTensor(y_train.to_numpy()))
    ## test data    
    class testData(Dataset):
        
        def __init__(self, X_data):
            self.X_data = X_data
            
        def __getitem__(self, index):
            return self.X_data[index]
            
        def __len__ (self):
            return len(self.X_data)
        
    
    test_data = testData(torch.FloatTensor(X_test.to_numpy()))
    test_data_full = testData(torch.FloatTensor(pd.concat([X_train, X_test],axis=0).to_numpy()))
    #test_data = testData(torch.FloatTensor(X_test.to_numpy()))
    
    
    class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in y_train])
    
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    test_full = DataLoader(dataset = test_data_full,batch_size=1)
    y_test_full = pd.concat([y_train,y_test],axis =0).to_numpy()
    
    
    class binaryClassification(nn.Module):
        def __init__(self):
            super(binaryClassification, self).__init__()
            # Number of input features is 12.
            self.layer_1 = nn.Linear(8, 32) 
            self.layer_2 = nn.Linear(32, 32)
            self.layer_out = nn.Linear(32, 1) 
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.batchnorm1 = nn.BatchNorm1d(32)
            self.batchnorm2 = nn.BatchNorm1d(32)
            
        def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.batchnorm1(x)
            x = self.relu(self.layer_2(x))
            x = self.batchnorm2(x)
            x = self.sigmoid(self.layer_out(x))
            #x = self.layer_out(x)
            
            return x
        
    model = binaryClassification()
    #model.to(device)
    print(model)
    
    criterion = nn.BCELoss()#BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for e in range(1, EPOCHS+1):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')
    
        
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            y_test_pred = model(X_batch)
            #y_test_pred = torch.sigmoid(y_test_pred)
            #y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_test_pred[0].detach().numpy())
            
    
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_pred_list = [1 if a>0.5 else 0 for a in y_pred_list]
    f1_ = f1_score(y_test, y_pred_list)
    print(confusion_matrix(y_test, y_pred_list))
    
    y_pred_list_full = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_full:
            #X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            #y_test_pred = torch.sigmoid(y_test_pred)
            #y_pred_tag = torch.round(y_test_pred)
            y_pred_list_full.append(y_test_pred[0].detach().numpy())
    
    y_pred_list_full = [a.squeeze().tolist() for a in y_pred_list_full]
    y_pred_list_full = [1 if a>0.5 else 0 for a in y_pred_list_full]
    f1 = f1_score(y_test_full, y_pred_list_full)
    print(confusion_matrix(y_test_full, y_pred_list_full))
    return f1_, f1

