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
import os
from scipy import stats
import random as r
import train as models
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

def cgc(path):
    '''
    Known Causes/Driver Genes
    Input: none
    Load a csv file previously downloaded
    output: Return the csv file
    '''
    dgenes = pd.read_csv(path,sep=',')
    dgenes['Tumour Types(Somatic)'] = dgenes['Tumour Types(Somatic)'].fillna(dgenes['Tumour Types(Germline)'])
    dgenes['y_out']=1
    dgenes = dgenes.iloc[:,[0,-1]]
    dgenes.rename(columns = {'Gene Symbol':'genes'}, inplace = True)
    return dgenes

def data_norm(data1):
    '''
    normalized data x- mean/sd
    input: dataset to be normalized
    output: normalized dataset

    '''
    data1o = np.zeros(data1.shape)
    data1o[:,0] = data1.iloc[:,0]

    for i in range(1,data1.shape[1]):
        nonzero = []
        for j in range(data1.shape[0]):
            if data1.iloc[j,i]!=0:
                nonzero.append(data1.iloc[j,i])
        for j in range(data1.shape[0]):
            if data1.iloc[j,i]!= 0:
                data1o[j,i] = (data1.iloc[j,i] - np.mean(nonzero))/np.sqrt(np.var(nonzero))

    data1o = pd.DataFrame(data1o)
    data1o.index = data1.index
    data1o.columns = data1.columns
    return data1o

def sim_load_vcf_to_h5(vcf_path,h5_path):
    '''
    transform data from vcf to h5
    input: paths
    output: new file on h5_path
    '''
    #download ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/
    #Reference: http://alimanfoo.github.io/2016/06/10/scikit-allel-tour.html
    #Example: http://alimanfoo.github.io/2017/06/14/read-vcf.html
    #vcf_path = 'C://Users//raque//Documents//GitHub//ParKCa//data_s//ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.vcf.gz'
    #h5_path = 'data_s//ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.h5'
    allel.vcf_to_hdf5(vcf_path, h5_path, fields='*', overwrite=True)

def sim_load_h5_to_PCA(h5_path):
    '''
    load dataset from h5 format file, remove non-informative columns,
    fit a PCA
    input: path file
    output:PCA coordenates
    '''
    callset = h5py.File(h5_path, mode='r')
    #Reference: http://alimanfoo.github.io/2015/09/28/fast-pca.html
    g = allel.GenotypeChunkedArray(callset['calldata/GT'])
    ac = g.count_alleles()[:]

    # remove singletons and multiallelic SNPs. Singletons are not informative for PCA,
    flt = (ac.max_allele() == 1) & (ac[:, :2].min(axis=1) > 1)
    gf = g.compress(flt, axis=0)
    # transform the genotype data into a 2-dimensional matrix where each cell has the number of non-reference alleles per call
    gn = gf.to_n_alt()

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

def sim_dataset(G0,lambdas,n_causes,n_units, randseed):
    '''
    calculate the target Y based on the simulated dataset

    input:
    G0: level 0 data
    lambdas: unknown groups
    n_causes and n_units: int, dimensions of the dataset
    output:
    G: G0 in pandas format with colnames that indicate if its a cause or not
    tc: causal columns
    y01: binary target

    '''
    np.random.seed(randseed)
    tc_ = npr.normal(loc = 0 , scale=0.5*0.5, size=int(n_causes*0.1))
    tc = np.hstack((np.repeat(0.0,n_causes-int(n_causes*0.1)),tc_))    #True causes
    tc.shuffle(tc)

    tau =  stats.invgamma(3,1).rvs(3, random_state = 99)
    sigma = np.zeros(n_units)
    sigma = [tau[0] if lambdas[j]==0 else sigma[j] for j in range(len(sigma))]
    sigma = [tau[1] if lambdas[j]==1 else sigma[j] for j in range(len(sigma))]
    sigma = [tau[2] if lambdas[j]==2 else sigma[j] for j in range(len(sigma))]
    y0 = np.array(tc).reshape(1,-1).dot(np.transpose(G0))
    l1 = lambdas.reshape(1,-1)
    y1 = (np.sqrt(np.var(y0))/np.sqrt(0.4))*(np.sqrt(0.4)/np.sqrt(np.var(l1)))*l1
    e = npr.normal(0,sigma,n_units).reshape(1,-1)
    y2 =  (np.sqrt(np.var(y0))/np.sqrt(0.4))*(np.sqrt(0.2)/np.sqrt(np.var(e)))*e
    y = y0 + y1 + y2
    p = 1/(1+np.exp(y0 + y1 + y2))

    y01 = np.zeros(len(p[0]))
    y01 = [npr.binomial(1,p[0][i],1)[0] for i in range(len(p[0]))]
    y01 = np.asarray(y01)
    G = add_colnames(G0,tc)
    return G, tc,y01

def add_colnames(data, truecauses):
    '''
    from matrix to pandas dataframe, adding colnames
    '''
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

def sim_genes_TGP(Fs, ps, n_hapmapgenes, n_causes, n_units, S, D, randseed):
    '''
    #Adapted from Deconfounder's authors
    generate the simulated data
    input:
        - Fs, ps, n_hapmapgenes: not used here
        - n_causes = integer
        - n_units = m (columns)
        - S: PCA output n x 2
    '''
    np.random.seed(randseed)

    S = expit(S)
    Gammamat = np.zeros((n_causes, 3))
    Gammamat[:,0] = 0.2*npr.uniform(size=n_causes) #0.45
    Gammamat[:,1] = 0.2*npr.uniform(size=n_causes) #0.45
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

def generate_samples(SIMULATIONS,n_units,n_causes):
    '''
    Input:
    SIMULATIONS: number of datasets to be produced
    n_units, n_causes: dimentions

    Output (pickle format):
    snp_simulated datasets
    y: output simulated and truecases for each datset are together in a single matrix

    Note: There are options to load the data from vcf format and run the pca
    Due running time, we save the files and load from the pca.txt file
    '''
    #ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/
    vcf_path = "data_s//ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.vcf.gz"
    h5_path = 'data_s//ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.h5'
    S = np.loadtxt('data_s//tgp_pca2.txt', delimiter=',')

    sim_y = []
    sim_tc = []
    for sim in range(SIMULATIONS):
        G0, lambdas = sim_genes_TGP([], [], 0 , n_causes, n_units, S, 3, sim )
        G1, tc, y01 = sim_dataset(G0,lambdas, n_causes,n_units,sim)
        G = add_colnames(G1,tc)
        del G0,G1

        G.to_pickle('data_s//snp_simulated1_'+str(sim)+'.txt')
        sim_y.append(y01)
        sim_tc.append(tc)
    sim_y = np.transpose(np.matrix(sim_y))
    sim_y = pd.DataFrame(sim_y)
    sim_y.columns = ['sim_'+str(sim) for sim in range(SIMULATIONS)]

    sim_tc = np.transpose(np.matrix(sim_tc))
    sim_tc = pd.DataFrame(sim_tc)
    sim_tc.columns = ['sim_'+str(sim) for sim in range(SIMULATIONS)]

    sim_y.to_pickle('data_s//snp_simulated1_y01.txt')
    sim_tc.to_pickle('data_s//snp_simulated1_truecauses.txt')

def join_simulation(path, version):
    '''
    the simulations were break down in several files;
    this function join the files from a same simulated dataset
    input: path files and dataset version
    output: a unique file with the cate of all treatments in a simulated dataset

    '''

    letter = ['a','b','c','d','e','f','g','h']
    files = []

    for l in letter:
        check = path+'cevae_output_sim'+str(version)+'_'+l+'.txt'
        if os.path.isfile(check):
            files.append(path+'cevae_output_sim'+str(version)+'_'+l+'.txt')

    sim = pd.read_pickle(files[0])
    if len(files) >= 1:
        for i in range(len(files)-1):
            #sim0 = sim0.iloc[0:3599,:]
            part = pd.read_pickle(files[i+1])
            sim = pd.concat([sim,part],axis = 0)

    sim.reset_index(inplace = True,drop = True)
    sim['cate'].fillna(0, inplace = True)
    return sim

def sim_level1data(simulations,tc,y01,roc_name):
    '''
    create a dataframe with the roc curves of the DA, and join in a single file the simulations results

    input:
        simulations: list with indices of simulated datasets
        tc: matrix with true causes. Col 1 has the true causes of simulation 1
        y01: matrix with targets. Col 1 has the targets of simulation 1
        roc_name: name file
    output: a file with the roc_curve data of the DA model is saved at roc_name path
    '''
    roc_table = pd.DataFrame(columns=['learners', 'fpr','tpr','auc'])
    for i in simulations:
        sim = 'sim_'+str(i)
        tc_sim1 = tc[sim]
        tc_sim1_bin = [1 if i != 0 else 0 for i in tc_sim1]
        y01_sim1 = y01[sim]

        train = pd.read_pickle('data_s\\snp_simulated1_'+str(i)+'.txt')
        coef, coef_continuos, roc, coln = models.deconfounder_PPCA_LR(np.asmatrix(train),train.columns,y01_sim1,sim,15,10)
        #Join CEVAE results
        cevae = join_simulation(path = 'results\\simulations\\',   version = i)
        roc_table = roc_table.append(roc,ignore_index=True)
        #Create dataset
        data = pd.DataFrame({'cevae':cevae['cate'],'coef':coef,'y_out':tc_sim1_bin, 'coef_c':coef_continuos,'y_c':tc_sim1})
        data.to_csv('results\\level1data_sim_'+str(i)+'.txt', sep=';')

    roc_table.to_pickle('results//'+roc_name+'.txt')
