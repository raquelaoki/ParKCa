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
from scipy.special import expit
import h5py
import sys
import allel #http://alimanfoo.github.io/2016/06/10/scikit-allel-tour.html

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
def sim_load_vcf_to_h5(vcf_path,h5_path):
    #download ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/
    #Reference: http://alimanfoo.github.io/2016/06/10/scikit-allel-tour.html
    #Example: http://alimanfoo.github.io/2017/06/14/read-vcf.html
    #Worked example: human 1000 genomes phase 3
    #vcf_path = 'C://Users//raque//Documents//GitHub//ParKCa//data_s//ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.vcf.gz'
    #h5_path = 'data_s//ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.h5'
    allel.vcf_to_hdf5(vcf_path, h5_path, fields='*', overwrite=True)

def sim_load_h5_to_PCA(h5_path):
    callset = h5py.File(h5_path, mode='r')

    #Reference: http://alimanfoo.github.io/2015/09/28/fast-pca.html
    #g = allel.GenotypeChunkedArray(callset['calldata']['genotype'])
    g = allel.GenotypeChunkedArray(callset['calldata/GT'])
    #g

    ac = g.count_alleles()[:]
    #ac

    # remove singletons and multiallelic SNPs. Singletons are not informative for PCA,
    #np.count_nonzero(ac.max_allele() > 1)
    #np.count_nonzero((ac.max_allele() == 1) & ac.is_singleton(1))
    flt = (ac.max_allele() == 1) & (ac[:, :2].min(axis=1) > 1)
    gf = g.compress(flt, axis=0)
    #gf
    # transform the genotype data into a 2-dimensional matrix where each cell has the number of non-reference alleles per call
    gn = gf.to_n_alt()
    #gn

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
    #more than 3 does not remove almost anything
    gnu = ld_prune(gn, size=500, step=200, threshold=.1, n_iter=3)

    #PCA
    k = 2
    coords1, model1 = allel.pca(gnu, n_components=k, scaler='patterson')
    np.savetxt('data_s//tgp_pca'+str(k)+'.txt', coords1, delimiter=',')
    return coords1

def sim_dataset(G0,lambdas,n_causes,n_units):
    tc_ = npr.normal(loc = 0 , scale=0.5*0.5, size=n_causes)
    #True causes
    tc = [i if i>np.quantile(tc_,0.99) else 0 for i in tc_]#truncate so only 1% are different from 0
    sigma = np.zeros(n_units)
    sigma = [1*1 if lambdas[j]==0 else sigma[j] for j in range(len(sigma))]
    sigma = [2*2 if lambdas[j]==1 else sigma[j] for j in range(len(sigma))]
    sigma = [5*5 if lambdas[j]==2 else sigma[j] for j in range(len(sigma))]
    y0 = np.array(tc).reshape(1,-1).dot(np.transpose(G0))
    y1 = 13*lambdas.reshape(1,-1)
    y2 = npr.normal(0,sigma,n_units).reshape(1,-1)
    y = y0 + y1 + y2
    p = 1/(1+np.exp(y0 + y1 + y2))
    print(np.var(y),np.var(y0),np.var(y1),np.var(y2))
    print('This should be 10%: ', np.var(y0)/np.var(y-y0))
    print('This should be 20%: ', np.var(y1)/np.var(y-y1))
    print('This should be 70%: ', np.var(y2)/np.var(y-y2))
    #del y0, y1, y2,y
    y01 = np.zeros(len(p[0]))
    y01 = [npr.binomial(1,p[0][i],1)[0] for i in range(len(p[0]))]
    y01 = np.asarray(y01)
    #568 1's
    print(sum(y01), len(y01))


    G = add_colnames(G0,tc)
    return G, tc,y01

def add_colnames(data, truecauses):
    colnames = []
    causes = 0
    noncauses = 0
    for i in range(len(truecauses)):
        if truecauses[i]>0:
            colnames.append('causal_'+str(causes))
            causes+=1
        else:
            colnames.append('noncausal_'+str(noncauses))
            noncauses+=1

    data = pd.DataFrame(data)
    data.columns = colnames
    return data


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

def sim_genes_TGP(Fs, ps, n_hapmapgenes, n_causes, n_units, S, D=3):
    '''
    generate the simulated data
    input:
        - Fs, ps, n_hapmapgenes: not used here
        - n_causes = integer
        - n_units = m (columns)
        - S: PCA output n x 2
    '''

    #Fs and ps [] (not used in this version)
    #n_hapmapgenes also not used
    #n_causes, n_units, S
    #pca = PCA(n_components=2, svd_solver='full')
    #S = expit(pca.fit_transform(hapmap_gene_clean))
    S = expit(S)
    Gammamat = np.zeros((n_causes, 3))
    Gammamat[:,0] = 0.45*npr.uniform(size=n_causes)
    Gammamat[:,1] = 0.45*npr.uniform(size=n_causes)
    Gammamat[:,2] = 0.05*np.ones(n_causes)
    S = np.column_stack((S[npr.choice(S.shape[0],size=n_units,replace=True),], \
        np.ones(n_units)))
    F = S.dot(Gammamat.T)
    #it was 2 instead of 1: goal is make SNPs binary
    G = npr.binomial(1, F)
    #unobserved group
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
