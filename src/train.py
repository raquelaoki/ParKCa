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





def deconfounder(train,colnames,y01,type1,k):

    #df = pd.DataFrame(np.arange(1,10).reshape(3,3))
    #arr = sparse.coo_matrix(([1,1,1], ([0,1,2], [1,2,0])), shape=(3,3))
    #df['newcol'] = arr.toarray().tolist()
    W,F = fm_MF(train,k)
    name_MF = type1+'_'+'MF'+'_'+str(k)
    causal_effect,roc, gamma,cil,cip  = check_save(W,train,colnames,y01,'MF',type1,k)
    df_ce = causal_effect
    df_roc = pd.DataFrame({name_MF:roc})
    df_gamma = pd.DataFrame({'model':[name_MF],'gamma':[gamma],'gamma_l':[cil],'gamma_u':[cip]})

    pca = fm_PCA(train,k)
    name_PCA = type1+'_'+'PCA'+'_'+str(k)
    causal_effect,roc, gamma,cil,cip = check_save(pca,train,colnames,y01,'PCA',type1,k)
    df_ce =pd.merge(df_ce, causal_effect,  how='left', left_on='genes', right_on = 'genes')
    df_roc[name_PCA]=roc
    aux = pd.DataFrame({'model':[name_PCA],'gamma':[gamma],'gamma_l':[cil],'gamma_u':[cip]})
    df_gamma = pd.concat([df_gamma,aux],axis=0)
    #df_gamma[name_PCA] = sparse.coo_matrix((gamma_ic),shape=(1,3)).toarray().tolist()

    ac = fm_A(train,k)
    name_A = type1+'_'+'A'+'_'+str(k)
    causal_effect,roc, gamma,cil,cip = check_save(pca,train,colnames,y01,'A',type1,k)
    df_ce =pd.merge(df_ce, causal_effect,  how='outer', left_on='genes', right_on = 'genes')
    df_roc[name_A] = roc
    aux = pd.DataFrame({'model':[name_A],'gamma':[gamma],'gamma_l':[cil],'gamma_u':[cip]})
    df_gamma = pd.concat([df_gamma,aux],axis=0)

    return df_ce, df_roc, df_gamma

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

def fm_PCA(train,k):
    '''
    PCA to extrac latent features
    Parameters:
        train: dataset
        k: latent Dimension
    Return:
        1 matrix
    '''
    X = StandardScaler().fit_transform(train)
    model = PCA(n_components=k)
    principalComponents = model.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)

    return principalDf

def fm_A(train,k):
    from keras.layers import Input, Dense
    from keras.models import Model
    '''
    Autoencoder to extrac latent features
    Parameters:
        train: dataset
        k: latent Dimension
        run: True/False
    Return:
        1 matrix
    References
    #https://www.guru99.com/autoencoder-deep-learning.html
    #https://blog.keras.io/building-autoencoders-in-keras.html
    '''
    x_train, x_test = train_test_split(train, test_size = 0.3,random_state = 22)
    print(x_train.shape, x_test.shape, train.shape)
    ii = x_train.shape[1]
    input_img = Input(shape=(ii,))
    encoding_dim = 20
    encoded = Dense(encoding_dim, activation='sigmoid')(input_img) #change relu
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(ii, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)


    autoencoder.compile(optimizer='sgd', loss='mean_squared_error')
    autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

    encoded_imgs = encoder.predict(train)
    return encoded_imgs

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

def predictive_check(X,Z):

    '''
    This function is agnostic to the method.
    Use a Linear Model X_m = f(Z), save the proportion
    of times that the pred(z_test)<X_m(test) for each feature m.
    Compare with the proportion of the null model mean(x_m(train)))<X_m(test)
    Create an Confidence interval for the null model, check if the average value
    across the predicted values using LM is inside this interval

    Sample a few columns (300 hundred?) to do this math

    Parameters:
        X: orginal features
        Z: latent (either the reconstruction of X or lower dimension)
    Return:
        v_obs values and result of the test
    '''
    #If the number of columns is too large, select a subset of columns instead
    if X.shape[1]>10000:
        X = X[:,np.random.randint(0,X.shape[1],10000)]

    v_obs = []
    v_nul = []
    for i in range(X.shape[1]):
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X[:,i], test_size=0.3)
        model = LinearRegression().fit(Z_train, X_train)
        X_pred = model.predict(Z_test)
        v_obs.append(np.less(X_test, X_pred).sum()/len(X_test))
        v_nul.append(np.less(X_test, X_train.mean(),).sum()/len(X_test))

    #Create the Confidence interval
    n = len(v_nul)
    m, se = np.mean(v_nul), np.std(v_nul)
    h = se * stats.t.ppf((1 + 0.95) / 2., n-1)
    if m-h<= np.mean(v_obs) and np.mean(v_obs) <= m+h:
        return np.mean(v_obs), m-h, m+h, True
    else:
        return round(np.mean(v_obs),4), round(m-h,4), round(m+h,4), False

def outcome_model(train,colnames , z, y01,colname1):
    '''
    Outcome Model + logistic regression
    I need to use less features for each model, so i can run several
    batches of the model using the latent features. They should account
    for all cofounders from the hole data

    pred: is the average value thought all the models

    parameters:
        train: dataset with original features jxv
        z: latent features, jxk
        y01: response, jx1

    return: list of significant coefs
    '''
    #if ac, change 25 to 9
    aux = train.shape[0]//9


    lim = 0
    col_new_order = []
    col_pvalue = []
    col_coef = []

    pred = []
    warnings.filterwarnings("ignore")


    #while flag == 0 and lim<=50:
    if train.shape[1]>aux:
        columns_split = np.random.randint(0,train.shape[1]//aux,train.shape[1] )

    #print('Aux value: ',aux)
    flag1 = 0
    for cs in range(0,train.shape[1]//aux):

        cols = np.arange(train.shape[1])[np.equal(columns_split,cs)]
        colnames_sub = colnames[np.equal(columns_split,cs)]
        col_new_order.extend(colnames_sub)
        X = pd.concat([pd.DataFrame(train[:,cols]),pd.DataFrame(z)], axis= 1)
        X.columns = range(0,X.shape[1])
        flag = 0
        lim = 0
        while flag==0 and lim <= 50:
            try:
                output = sm.Logit(y01, X).fit(disp=0)
                pred.append(output.predict(X))
                flag = 1
                flag1 = 1
            except:
                flag = 0
                lim = lim+1
                print('--------- Trying again----------- ',colname1, aux,cs)

        if flag == 1:
            col_pvalue.extend(output.pvalues[0:(len(output.pvalues)-z.shape[1])])
            col_coef.extend(output.params[0:(len(output.pvalues)-z.shape[1])])
        else:
            col_pvalue.extend(np.repeat(0,len(colnames_sub)))
            col_coef.extend(np.repeat(0,len(colnames_sub)))

    warnings.filterwarnings("default")
    #prediction only for the ones with models that converge
    pred1 =  np.mean(pred,axis = 0)
    resul =  pd.concat([pd.DataFrame(col_new_order),pd.DataFrame(col_pvalue), pd.DataFrame(col_coef)], axis = 1)
    resul.columns = ['genes',colname1+'_pvalue',colname1+'_coef']
    if flag1 == 0:
        output = []
        pred1 = []
    return resul, pred1

def outcome_model_ridge(train, colnames,x,y01,colnames1):
    #calculate IC with bootstrap
    import scipy.stats as st
    #st.norm.ppf(.95)#1.6448536269514722
    #st.norm.cdf(1.64)#0.94949741652589625
    alphas = np.linspace(.00001, 2, 1)
    coef = []
    pred = []
    sim = 500
    for i in range(sim):
        rows = np.random.randint(0,train.shape[0],size=int(train.shape[0]*0.9))
        ridge = linear_model.RidgeClassifierCV(alphas = alphas, cv =5, normalize = True)
        ridge.fit(train[rows], y01[rows])
        coef.append(ridge.coef_[0])
        pred.append(ridge.predict(train))
    coef2 = np.array(coef)
    coef_mean = coef2.mean(axis=0)
    coef_var = coef2.var(axis=0)
    z_values = coef_mean/(np.sqrt(coef_var/sim))
    coef_pvalues = st.norm.cdf(z_values)
    coef_low = coef_mean - 1.96*np.sqrt(coef_var/sim)
    coef_up= coef_mean + 1.96*np.sqrt(coef_var/sim)
    
    pred = np.array(pred)
    pred = pred.mean(axis=0)
    
    resul = pd.DataFrame({'genes':colnames,colname1+'_pvalue': coef_pvalues,colname1+'_coef':coef_mean })
    
    extra = pd.DataFrame({'genes':colnames,colname1+'_icl': coef_low,colname1+'_icu':coef_up })
    
    return resul, pred, extra