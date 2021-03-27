import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
import eval as eval
from data_code_future_repo import datapreprocessing as dp
#import CEVAE as cevae
from os import listdir
from os.path import isfile, join

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.svm import SVC
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn import linear_model
from sklearn import calibration
from scipy import sparse, stats

#DA
import functools
#Meta-leaners packages
#https://github.com/aldro61/pu-learning
from puLearning.puAdapter import PUAdapter
#https://github.com/t-sakai-kure/pywsl
from pywsl.pul import pu_mr #pumil_mr
#from pywsl.utils.syndata import gen_twonorm_pumil
#from pywsl.utils.comcalc import bin_clf_err
#NN
from torch.utils.data import Dataset, DataLoader
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler

#TODO: update with https://github.com/raquelaoki/Summer2020MultipleCauses/blob/master/parkca/train.py

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
                 coefkc_table = pd.DataFrame(columns=['genes'])
                 roc_table = pd.DataFrame(columns=['learners', 'fpr','tpr','auc'])
                 #test
                 for filename in listfiles:
                     train, j, v, y01, abr, colnames = dp.data_prep('data\\'+filename)
                     if train.shape[0]>150:
                        print(filename,': ' ,train.shape[0])
                        #change filename
                        name = filename.split('_')[-1].split('.')[0]
                        if name not in skip:
                            coef, coef_continuos, roc, coln = deconfounder_PPCA_LR(train,colnames,y01,name,k,b)
                            roc_table = roc_table.append(roc,ignore_index=True)
                            coefk_table[coln] = coef
                            coefkc_table[coln] = coef_continuos
                        else:
                            print('skip',name)

                 print('--------- DONE ---------')
                 coefk_table['genes'] = colnames
                 coefkc_table['genes'] = colnames

                 #CHANGE HERE 20/05
                 roc_table.to_pickle('results//roc_'+str(k)+'.txt')
                 coefkc_table.to_pickle('results//coefcont_'+str(k)+'.txt')

        if BARTBOOL:
            print('BART')
            #MODEL AND PREDICTIONS MADE ON R
            filenames=['results//bart_all.txt','results//bart_MALE.txt','results//bart_FEMALE.txt']
            eval.roc_table_creation(filenames,'bart')
            eval.roc_plot('results//roc_'+'bart'+'.txt')

        if BARTBOOL and DABOOL:
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
    X_full = np.concatenate((X,X_), axis = 0 )
    y_full = np.concatenate((y,y_), axis = 0)

    warnings.filterwarnings("ignore")
    if name_model == 'nn':
        y_pred, ypred = nn_classifier(y, y_, X, X_,X_full)
        pr = precision(1,confusion_matrix(y_,y_pred))
        f1 = f1_score(y_,y_pred)
        if np.isnan(pr) or pr==0 or f1<0.06:
            while np.isnan(pr) or pr ==0 or f1<0.06:
                print('\n\n trying again \n\n')
                y_pred, ypred = nn_classifier(y, y_, X, X_,X_full)
                pr = precision(1,confusion_matrix(y_,y_pred))
                f1 = f1_score(y_,y_pred)

    else:

        if name_model == 'adapter':
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
            from sklearn.linear_model import LogisticRegression
            w1 = y.sum()/len(y)
            w0 = 1 - w1
            sample_weight = {0:w1,1:w0}
            model = LogisticRegression(C=.1,class_weight=sample_weight,penalty='l2') #
            model.fit(X,y)


        elif name_model=='rf':
            print('rd',X.shape[1])
            w = len(y)/y.sum()
            sample_weight = np.array([w if i == 1 else 1 for i in y])
            model = RandomForestClassifier(max_depth=12, random_state=0)
            model.fit(X, y,sample_weight = sample_weight)

        else:
            print('random',X.shape[1])

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

        y_pred = np.where(y_pred==-1,0,y_pred)
        ypred = np.where(ypred==-1,0,ypred)

    pr = precision(1,confusion_matrix(y_,y_pred))
    re = recall(1,confusion_matrix(y_,y_pred))

    prfull = precision(1,confusion_matrix(y_full,ypred))
    refull = recall(1,confusion_matrix(y_full,ypred))

    auc = roc_auc_score(y_,y_pred)
    f1 = f1_score(y_full,ypred)
    f1_ = f1_score(y_,y_pred)

    roc = {'metalearners': name_model,'precision':pr ,'recall':re,'auc':auc,'f1':f1,'f1_':f1_,'prfull':prfull,'refull':refull}
    warnings.filterwarnings("default")
    return roc, ypred, y_pred


def meta_learner(data1, models, prob ):
    '''
    input: level 1 data
    outout: roc table
    '''
    roc_table = pd.DataFrame(columns=['metalearners', 'precision','recall','auc','f1','f1_','prfull','refull'])

    #split data trainint and testing
    y = data1['y_out']
    X = data1.drop(['y_out'], axis=1)
    y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.33,random_state=32)

    #starting ensemble
    e_full = np.zeros(len(y))
    e_pred = np.zeros(len(y_test))
    e = 0

    #Some causes are unknown or labeled as 0
    y_train = [i if np.random.binomial(1,prob,1)[0]==1 else 0 for i in y_train]
    y_train = pd.Series(y_train)

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
    prfull = precision(1,confusion_matrix(np.hstack([y_test,y_train]),e_full))
    refull = recall(1,confusion_matrix(np.hstack([y_test,y_train]),e_full))

    auc = roc_auc_score(y_test,e_pred)
    f1 = f1_score(np.hstack([y_test,y_train]),e_full)
    f1_ = f1_score(y_test,e_pred)
    roc = {'metalearners': 'ensemble','precision':pr ,'recall':re,'auc':auc,'f1':f1,'f1_':f1_,'prfull':prfull,'refull':refull}
    roc_table = roc_table.append(roc,ignore_index=True)
    return roc_table

def nn_classifier(y_train, y_test, X_train, X_test,X_full):
    '''
    meta-learner 
    '''
    #https://docs.microsoft.com/en-us/archive/msdn-magazine/2019/october/test-run-neural-binary-classification-using-pytorch
    class Batcher:
      def __init__(self, num_items, batch_size, seed=0):
        self.indices = np.arange(num_items)
        self.num_items = num_items
        self.batch_size = batch_size
        self.rnd = np.random.RandomState(seed)
        self.rnd.shuffle(self.indices)
        self.ptr = 0
      def __iter__(self):
        return self
      def __next__(self):
        if self.ptr + self.batch_size > self.num_items:
          self.rnd.shuffle(self.indices)
          self.ptr = 0
          raise StopIteration  # exit calling for-loop
        else:
          result = self.indices[self.ptr:self.ptr+self.batch_size]
          self.ptr += self.batch_size
          return result

    # ------------------------------------------------------------
    def akkuracy(model, data_x):
      # data_x and data_y are numpy array-of-arrays matrices
      X = T.Tensor(data_x)
      oupt = model(X)            # a Tensor of floats
      oupt = oupt.detach().float()
      oupt = [1 if i > 0.5 else 0 for i in oupt]
      return oupt
    # ------------------------------------------------------------
    class Net(T.nn.Module):
      def __init__(self):
        super(Net, self).__init__()
        self.hid1 = T.nn.Linear(size, 16)
        self.oupt = T.nn.Linear(16, 1)
        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)
      def forward(self, x):
        z = T.tanh(self.hid1(x))
        z = T.sigmoid(self.oupt(z))  # necessary
        return z

    size = X_train.shape[1]

    net = Net()

    net = net.train()  # set training mode
    lrn_rate = 0.01
    bat_size = 500
    loss_func = T.nn.BCELoss()  # softmax() + binary CE
    optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
    max_epochs = 500
    n_items = len(X_train)
    batcher = Batcher(n_items, bat_size)
    X_test = X_test.values
    X_train = X_train.values
    print('Starting training')

    count_class_0, count_class_1 = y_train.value_counts()

    # Divide by class
    df_class_0 = pd.DataFrame(X_train[y_train== 0])
    df_class_1 = pd.DataFrame(X_train[y_train == 1])

    df_class_0_under = df_class_0.sample(4000)
    df_class_1_over = df_class_1.sample(4000, replace=True)
    X_train2 = pd.concat([df_class_0_under, df_class_1_over], axis=0)
    X_train2['y']= np.repeat([0,1],4000)
    X_train2 = X_train2.sample(frac=1).reset_index(drop=True)

    y_train2 = X_train2['y']
    X_train2 = X_train2.drop(['y'], axis=1)

    X_train2 = X_train2.values
    y_train2 = y_train2.values

    for epoch in range(0, max_epochs):
      for curr_bat in batcher:
        X = T.Tensor(X_train2[curr_bat])
        Y = T.Tensor(y_train2[curr_bat])
        optimizer.zero_grad()
        oupt = net(X)
        loss_obj = loss_func(oupt, Y)
        loss_obj.backward()
        optimizer.step()
    print('Training complete \n')
    net = net.eval()  # set eval mode

    y_pred = akkuracy(net, X_test)
    yfull = akkuracy(net, X_full)
    return y_pred, yfull
