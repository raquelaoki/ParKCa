import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy.random as npr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import sparse

def load_GE(filename1, filename2):
    '''
    Function to load gene expression and clinical data complete
    input: 
        - filename1: gene expression data all
        - filename2: clinical data all 
    output:
        - pre-processed data
    '''
    df_ge = pd.read_csv(filename1,sep=';')
    df_cl = pd.read_csv(filename2,sep=';')
    #print(df_ge.shape,df_cl.shape)
    df_ge = df_ge.drop('patients2',axis=1)
    #using the log of gene expression
    columns = df_ge.columns
    columns = columns.drop('patients')
    for col in columns:
         df_ge[col] = np.log(df_ge[col]+1)

    #removing columns not important
    df_cl = df_cl.drop(['race','ethnicity','vital_status','tumor_status','aux'],axis=1)
    #rename columns and  binary columns
    df_cl.rename(columns={"new_tumor_event_dx_indicator": "y"},inplace=True)
    #get dummy
    columns_to_dummy = df_cl.columns_split
    columns_to_dummy = columns_to_dummy.drop('patients')
    for col in columns_to_dummy:
        just_dummies = pd.get_dummies(df_cl[col])
        df_cl = pd.concat([df_cl, just_dummies], axis=1)
        df_cl = df_cl.drop(col,axis=1)

    df_ge.to_csv('data\\train_ge.txt', sep=';', index = False)
    df_cl.to_csv('data\\train_cl.txt', sep=';', index = False)
    return df_ge,df_cl

def read_GE(filename1, filename2):
    '''
    parameters:
        filename1 and filename2: address of ge and cl data
    return:
        train: training set read to use
        y01: classifications
        abr: cancer types
        gender: gender
        colnames: gene names
    '''
    df_ge = pd.read_csv(filename1,sep=';')
    df_cl = pd.read_csv(filename2,sep=';')

    df_ge.sort('patients',inplace=True)
    df_cl.sort('patients',inplace=True)
    if not np.array_equal(df_ge['patients'],df_ge['patients']):
        print('Error: order is different')

    y01 = np.array(df_cl['y'])
    abr = np.array(df_cl['abr'])
    gender = np.array(df_cl['gender'])
    df_cl.drop(['y','abr'],axis=1,inplace=True)
    train = df_ge.drop('patients',axis=1)
    del df_ge
    del df_cl
    colnames = train.columns
    train = np.matrix(train)
    return train,colnames, y01, [abr,gender]

def data_prep(filename):
    '''
    parameters:
        data: full dataset
    return:
        train: training set withouht these elements
        j,v: dimensions
        y01: classifications
        abr: cancer types
        colnames: gene names
    '''
    data = pd.read_csv(filename,sep=';')
    data = data.reset_index(drop=True)
    '''Organizing columns names'''
    remove = data.columns[[0,1,2]]
    y = data.columns[1]
    y01 = np.array(data[y])
    abr = np.array(data[data.columns[2]])
    train = data.drop(remove, axis = 1)
    colnames = train.columns
    train = np.matrix(train)

    j, v = train.shape

    return train, j, v, y01,  abr, colnames


#Code bellow adapted from Yixin Wang
def sim_genes_BN(Fs, ps, n_hapmapgenes, n_causes, n_units, D=3):
    '''
    inputs: 
        - Fs: matrix (format?)
        - ps: matrix (format?)
        - n_hapmapgenes: possible causes
        - n_causes: int/col
        - n_units:  size/row
    output:
        - G: 
        - lambdas: 
    '''
    
    idx = npr.randint(n_hapmapgenes, size = n_causes)
    p = ps[idx]
    F = Fs[idx]
    Gammamat = np.zeros((n_causes, D))
    for i in range(D):
        Gammamat[:,i] = npr.beta((1-F)*p/F, (1-F)*(1-p)/F)
    S = npr.multinomial(1, (60/210, 60/210, 90/210), size = n_units)
    F = S.dot(Gammamat.T)
    G = npr.binomial(2, F)
    lambdas = KMeans(n_clusters=3, random_state=123).fit(S).labels_
    sG = sparse.csr_matrix(G)
    return G, lambdas

def sim_genes_TGP(Fs, ps, n_hapmapgenes, n_causes, n_units, hapmap_gene_clean, D=3):
    pca = PCA(n_components=2, svd_solver='full')
    S = expit(pca.fit_transform(hapmap_gene_clean))
    Gammamat = np.zeros((n_causes, 3))
    Gammamat[:,0] = 0.45*npr.uniform(size=n_causes)
    Gammamat[:,1] = 0.45*npr.uniform(size=n_causes)
    Gammamat[:,2] = 0.05*np.ones(n_causes)
    S = np.column_stack((S[npr.choice(S.shape[0],size=n_units,replace=True),], \
        np.ones(n_units)))
    F = S.dot(Gammamat.T)
    G = npr.binomial(2, F)
    lambdas = KMeans(n_clusters=3, random_state=123).fit(S).labels_
    sG = sparse.csr_matrix(G)
    return G, lambdas

def sim_genes_PSD(Fs, ps, n_hapmapgenes, n_causes, n_units, D=3):
    alpha = 0.5
    idx = npr.randint(n_hapmapgenes, size = n_causes)
    p = ps[idx]
    F = Fs[idx]
    Gammamat = np.zeros((n_causes, D))
    for i in range(D):
        Gammamat[:,i] = npr.beta((1-F)*p/F, (1-F)*(1-p)/F)
    S = npr.dirichlet((alpha, alpha, alpha), size = n_units)
    F = S.dot(Gammamat.T)
    G = npr.binomial(2, F)
    lambdas = KMeans(n_clusters=3, random_state=123).fit(S).labels_
    sG = sparse.csr_matrix(G)
    return G, lambdas

def sim_genes_SP(Fs, ps, n_hapmapgenes, n_causes, n_units, D=3):
    a = 0.1
    # simulate genes
    Gammamat = np.zeros((n_causes, 3))
    Gammamat[:,0] = 0.45*npr.uniform(size=n_causes)
    Gammamat[:,1] = 0.45*npr.uniform(size=n_causes)
    Gammamat[:,2] = 0.05*np.ones(n_causes)
    S = npr.beta(a, a, size=(n_units, 2))
    S = np.column_stack((S, np.ones(n_units)))
    F = S.dot(Gammamat.T)
    G = npr.binomial(2, F)
    lambdas = KMeans(n_clusters=3, random_state=123).fit(S).labels_
    sG = sparse.coo_matrix(G)
    return G, lambdas