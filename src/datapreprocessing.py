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
    aplication
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
    aplication
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
    aplication
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


#simulation
#Reference: http://alimanfoo.github.io/2016/06/10/scikit-allel-tour.html
#Example: http://alimanfoo.github.io/2017/06/14/read-vcf.html
#Worked example: human 1000 genomes phase 3

vcf_path = 'C://Users//raque//Documents//GitHub//ParKCa//data_s//ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.vcf.gz'
import sys
import allel
import zarr
import numcodecs
callset = allel.read_vcf(vcf_path, fields=['numalt'], log=sys.stdout)
numalt = callset['variants/numalt']
count_numalt = np.bincount(numalt)
zarr_path = 'C://Users//raque//Documents//GitHub//ParKCa//data_s//ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.zarr'
#takes time: 3:43 - 4:15 (?) 
allel.vcf_to_zarr(vcf_path, zarr_path, fields='*', alt_number=np.max(numalt), log=sys.stdout,
                  compressor=numcodecs.Blosc(cname='zstd', clevel=1, shuffle=False))

#checking
callset_h1k = zarr.open_group(zarr_path, mode='r')
callset_h1k

pos = allel.SortedIndex(callset_h1k['/variants/POS'])
pos

#4:26 - 4:47
h5_path = 'C://Users//raque//Documents//GitHub//ParKCa//data_s//ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.h5'
allel.vcf_to_hdf5(vcf_path, h5_path, fields='*', overwrite=True)

import h5py
callset = h5py.File(h5_path, mode='r')



#LOAD h5 
#PCA http://alimanfoo.github.io/2015/09/28/fast-pca.html
#g = allel.GenotypeChunkedArray(callset['calldata']['genotype'])
g = allel.GenotypeChunkedArray(callset['calldata/GT'])
g

ac = g.count_alleles()[:]
ac

# remove singletons and multiallelic SNPs. Singletons are not informative for PCA,
np.count_nonzero(ac.max_allele() > 1)
np.count_nonzero((ac.max_allele() == 1) & ac.is_singleton(1))
flt = (ac.max_allele() == 1) & (ac[:, :2].min(axis=1) > 1)
gf = g.compress(flt, axis=0)
gf

# transform the genotype data into a 2-dimensional matrix where each cell has the number of non-reference alleles per call
gn = gf.to_n_alt()
gn

#Removing correlated features (LD pruning): each SNP is a feature, SNPs tend to be correlated
#It takes a while 5:15-
def ld_prune(gn, size, step, threshold=.1, n_iter=1):
    for i in range(n_iter):
        loc_unlinked = allel.locate_unlinked(gn, size=size, step=step, threshold=threshold)
        n = np.count_nonzero(loc_unlinked)
        n_remove = gn.shape[0] - n
        print('iteration', i+1, 'retaining', n, 'removing', n_remove, 'variants')
        gn = gn.compress(loc_unlinked, axis=0)
    return gn

gnu = ld_prune(gn, size=500, step=200, threshold=.1, n_iter=5)



#PCA 
coords1, model1 = allel.pca(gnu, n_components=2, scaler='patterson')




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