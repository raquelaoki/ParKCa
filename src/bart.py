import pandas as pd
import numpy as np
import warnings
from bartpy.sklearnmodel import SklearnModel

class BART():
    def __init__(self, X_train, X_test, y_train, y_test):
        super(BART, self).__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        print('Running BART')

    def Find_Optimal_Cutoff(self, target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------
        list type, with optimal cutoff value
        https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
        """
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
        return list(roc_t['threshold'])

    def fit(self, n_trees=50, n_burn=100):
        model = SklearnModel(n_trees=n_trees, n_burn=n_burn, n_chains=1, n_jobs=1)
        model.fit(self.X_train, self.y_train)
        y_train_pred = model.predict(self.X_train)  # [:,0:1000] Make predictions on the train set
        y_test_pred = model.predict(self.X_test)  # [:,0:1000] Make predictions on the train set
        thhold = self.Find_Optimal_Cutoff(self.y_train, y_train_pred)
        y_train_pred01 = [0 if item < thhold else 1 for item in y_train_pred]
        y_test_pred01 = [0 if item < thhold else 1 for item in y_test_pred]
        print('... Leaner Evaluation:')
        print('... Training set: F1 - ', f1_score(self.y_train, y_train_pred01))
        print('...... confusion matrix: ', confusion_matrix(self.y_train, y_train_pred01).ravel())

        print('... Testing set: F1 - ', f1_score(self.y_test, y_test_pred01))
        print('...... confusion matrix: ', confusion_matrix(self.y_test, y_test_pred01).ravel())
        assert isinstance(model, object)
        self.model = model


    def cate(self, TreatmentColumns, boostrap=False, b=30):
        # TODO: implement cate
        print('CATE In progress')
        X = np.concatenate([self.X_train, self.X_test], axis=0)
        # y = np.concatenate([self.y_train, self.y_test], axis=0)
        y_pred_full = self.model.predict(X)
        bart_cate = np.zeros(len(TreatmentColumns))

        for t, treat in enumerate(TreatmentColumns):
            print('Col', i)
            Xi = X.copy()
            Xi[:, treat] = 1 - Xi[:, treat]
            y_pred_fulli = self.model.predict(Xi)
            Ot1 = y_pred_full[X[:, treat] == 1]
            Ot0 = y_pred_full[X[:, treat] == 0]
            It1 = y_pred_fulli[Xi[:, treat] == 1]
            It0 = y_pred_fulli[Xi[:, treat] == 0]
            print(It1, It0, len(Y_pred_fulli))
            print(Ot1, Ot0, len(y_pred_full))
            print('treat filter', X[0:1, treat])
            print('treat filter', Xi[0:1, treat])
            # assert len(It1) + len(It0) == X_pred_fulli.shape[0], 'CATE: Wrong Dimensions'
            # assert len(Ot1) + len(Ot0) == len(y_pred_full), 'CATE: Wrong Dimensions'
            bart_cate[t] = np.concatenate([Ot1, It1], 0).mean() - np.concatenate([Ot0, It0], 0).mean()
        return bart_cate

    def boostrap_cate(self, dif, b=30):
        # TODO: implemente bootstrap cate
        print('TODO',b)
