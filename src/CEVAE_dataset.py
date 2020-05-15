import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

#0-200: ok


class DATA(object):
    def __init__(self, ncol, data_path, y01, replications = 20): #path_data="datasets/IHDP/csv",  replications=10
        #self.data_s = data_path
        self.data = np.matrix(pd.read_pickle(data_path))
        self.ncol = ncol
        self.y01 = y01
        self.replications = replications
        # which features are binary
        self.binfeats = list(range(self.ncol))#[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
        self.contfeats = []#[i for i in range(25) if i not in self.binfeats]

    def __iter__(self):
        for i in range(self.replications):
            tcol = i
            #data = self.data_s #np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            data = self.data
            #y = self.y01[:,self.tcol]
            #t, x = data[:,self.tcol], np.delete(data,self.tcol,1)
            t, y , x = data[:,tcol], self.y01, np.delete(data,tcol,1)
            yield (x, t, y)#, (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in range(self.replications):
            data = self.data
            tcol = i
            #y = self.y01[:,self.tcol]
            #t, x = data[:,self.tcol],  np.delete(data,self.tcol,1)
            t, y , x = data[:,tcol], np.asmatrix(self.y01).reshape(-1,1), np.delete(data,tcol,1)
            #t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            #mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]

            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.33, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.33, random_state=1)
            train = (x[itr], t[itr], y[itr])#, (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva])#, (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite])#, (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, self.contfeats, self.binfeats
