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
from sklearn import calibration

from scipy.stats import gamma
from scipy import sparse, stats

import statsmodels.discrete.discrete_model as sm

#from tensorflow.keras import optimizers
import tensorflow as tf #.compat.v2
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
tf.enable_eager_execution()
import functools


#libraries for BART
from bartpy.sklearnmodel import SklearnModel
import pickle 

def BART(train, colnames,y01,name,load,filename):
    #   
    '''
    CATE: using a subset of the traning set to speed up
    '''

    X, X_, y, y_= train_test_split(train,y01, test_size=0.33,random_state=42)
    if not load: 
        #it takes a long time to run
        model = SklearnModel(n_samples=1000, n_burn=200, n_trees=50, n_chains=1, n_jobs=-1, store_in_sample_predictions=False) 
        model.fit(np.array(X), y) # Fit the model
        # save the model to disk
        pickle.dump(model, open(filename, 'wb'))
    else: 
        # load the model from disk
        model = pickle.load(open(filename, 'rb'))
        result = model.score(X_test, Y_test)

    '''
    y_pred = model.predict(X_) 
    cate = []
    for i in rante(len(colnames)):
        X_inter = X_test.copy()
        X_inter[:,i] = 0 
        y_inter = model.predict(X_inter)
        #cate = 
    
    
        ROC: split training/testing set and return results for testing set
    '''
    return #cate, roc, filename



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
#it works, but all predictions where the same
def fm_PMF(train,k):
    rating = pd.DataFrame(train)
    tab_colname = pd.DataFrame({'idcol':rating.columns,'gene':colnames.values.tolist() })
    #rating.iloc[0:5,0:2]
    rating['id'] = rating.index
    rating1 = pd.melt(rating,id_vars='id',var_name='genes', value_name='values')
    #rating.id = rating.id + 1
    #rating.genes = rating.genes + 1 
    #shape: 20166364
    print(rating1.shape)    
    rating2 = rating1.to_numpy() #
    print(rating2.shape)
    del rating, rating1
    #filename = "data\\tcga_train_gexpression_cgc_7k.txt" #_2
    train, test = train_test_split(rating2, test_size=0.2)  # spilt_rating_dat(ratings)
    #epsilon, _lambda, momentum, maxepoch, num_batches, batch_size    
    #("num_batches", 10), ("batch_size", 1000)
    pmf = models.PMF(num_feat=k,_lambda=0.1, maxepoch = 6, momentum = 0.1,epsilon=1)#.fit(train, test)
    pmf.fit(train, test)
    w = pmf.w_Item
    u = pmf.w_User
    x_gen = []
    for i in range(train.shape[0]):
        x_gen.append(pmf.predict(i))
        
    #models.PMF(num_feat=k,_lambda=0.1) RMSE 2.802363 
    #models.PMF(num_feat=k,_lambda=0.3) RMSE 2.802346
    #models.PMF(num_feat=k,_lambda=0.8) RMSE 2.802315
    #momentum = from 0.8 to 0.5: no difference 
    #momentum from 0.8 to 1.5: RMSE increase and explode
    #epsilon from 1 to 0.5: no differnce 
    #epsilon from 1 to 2: no difference 
    return w, u, np.array(x_gen)

class PMF(object):
    #reference: https://github.com/fuhailin/Probabilistic-Matrix-Factorization
    def __init__(self, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=10, batch_size=1000):
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)

        self.w_Item = None  # Item feature vectors
        self.w_User = None  # User feature vectors

        self.rmse_train = []
        self.rmse_test = []

    # ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********#
    # ***************** train_vec=TrainData, test_vec=TestData*************#
    def fit(self, train_vec, test_vec):
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:, 2])  # 评分平均值

        pairs_train = train_vec.shape[0]  # traindata 中条目数
        pairs_test = test_vec.shape[0]  # testdata中条目数

        # 1-p-i, 2-m-c
        num_user = int(max(np.amax(train_vec[:, 0]), np.amax(test_vec[:, 0]))) + 1  # 第0列，user总数
        num_item = int(max(np.amax(train_vec[:, 1]), np.amax(test_vec[:, 1]))) + 1  # 第1列，movie总数

        incremental = False  # 增量
        if ((not incremental) or (self.w_Item is None)):
            # initialize
            self.epoch = 0
            self.w_Item = 0.1 * np.random.randn(num_item, self.num_feat)  # numpy.random.randn 电影 M x D 正态分布矩阵
            self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)  # numpy.random.randn 用户 N x D 正态分布矩阵

            self.w_Item_inc = np.zeros((num_item, self.num_feat))  # 创建电影 M x D 0矩阵
            self.w_User_inc = np.zeros((num_user, self.num_feat))  # 创建用户 N x D 0矩阵

        while self.epoch < self.maxepoch:  # 检查迭代次数
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])  # 根据记录数创建等差array
            np.random.shuffle(shuffled_order)  # 用于将一个列表中的元素打乱

            # Batch update
            for batch in range(self.num_batches):  # 每次迭代要使用的数据量
                # print "epoch %d batch %d" % (self.epoch, batch+1)

                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])  # 本次迭代要使用的索引下标

                batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_User[batch_UserID, :],
                                              self.w_Item[batch_ItemID, :]),
                                  axis=1)  # mean_inv subtracted # np.multiply对应位置元素相乘

                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                # Compute gradients
                Ix_User = 2 * np.multiply(rawErr[:, np.newaxis], self.w_Item[batch_ItemID, :]) \
                       + self._lambda * self.w_User[batch_UserID, :]
                Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis], self.w_User[batch_UserID, :]) \
                       + self._lambda * (self.w_Item[batch_ItemID, :])  # np.newaxis :increase the dimension

                dw_Item = np.zeros((num_item, self.num_feat))
                dw_User = np.zeros((num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_Item[batch_ItemID[i], :] += Ix_Item[i, :].astype(float)
                    dw_User[batch_UserID[i], :] += Ix_User[i, :].astype(float)

                # Update with momentum
                self.w_Item_inc = self.momentum * self.w_Item_inc + self.epsilon * dw_Item / self.batch_size
                self.w_User_inc = self.momentum * self.w_User_inc + self.epsilon * dw_User / self.batch_size

                self.w_Item = self.w_Item - self.w_Item_inc
                self.w_User = self.w_User - self.w_User_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(train_vec[:, 0], dtype='int32'), :],
                                                  self.w_Item[np.array(train_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    obj = np.linalg.norm(rawErr) ** 2 \
                          + 0.5 * self._lambda * (np.linalg.norm(self.w_User) ** 2 + np.linalg.norm(self.w_Item) ** 2)

                    self.rmse_train.append(np.sqrt(obj / pairs_train))

                # Compute validation error
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(test_vec[:, 0], dtype='int32'), :],
                                                  self.w_Item[np.array(test_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - test_vec[:, 2] + self.mean_inv
                    self.rmse_test.append(np.linalg.norm(rawErr) / np.sqrt(pairs_test))

                    # Print info
                    if batch == self.num_batches - 1:
                        print('Training RMSE: %f, Test RMSE %f' % (self.rmse_train[-1], self.rmse_test[-1]))

    def predict(self, invID):
        return np.dot(self.w_Item, self.w_User[int(invID), :]) + self.mean_inv  # numpy.dot 点乘

    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)

    def topK(self, test_vec, k=10):
        inv_lst = np.unique(test_vec[:, 0])
        pred = {}
        for inv in inv_lst:
            if pred.get(inv, None) is None:
                pred[inv] = np.argsort(self.predict(inv))[-k:]  # numpy.argsort索引排序

        intersection_cnt = {}
        for i in range(test_vec.shape[0]):
            if test_vec[i, 1] in pred[test_vec[i, 0]]:
                intersection_cnt[test_vec[i, 0]] = intersection_cnt.get(test_vec[i, 0], 0) + 1
        invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype='int32'))

        precision_acc = 0.0
        recall_acc = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k)
            recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])

        return precision_acc / len(inv_lst), recall_acc / len(inv_lst)

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
