import pandas as pd
import numpy as np
import warnings

import tensorflow.compat.v1 as tf
from tensorflow.keras import optimizers
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd  # conda install -c conda-forge tensorflow-probability
tf.disable_v2_behavior()
tf.enable_eager_execution()

class deconfounder_algorithm():
    def __init__(self, X_train, X_test, y_train, y_test, k=5):
        super(deconfounder_algorithm, self).__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        X = np.concatenate([self.X_train, self.X_test], axis=0)
        y = np.concatenate([self.y_train, self.y_test], axis=0)
        self.X = X
        self.y = y
        self.n = self.X.shape[0]
        self.ncol = self.X.shape[1]
        self.k = k
        print('Running DA')

    def fit(self, b=100, holdout_prop=0.2, alpha=0.05):
        """
        Implementation of the Deconfounder Algorthm with Prob PCA and Logistic Regression
        Contains: Prob PCA function, Predictive Check and Outcome Model
        input:
        - colnames or possible causes
        - k: dimension of latent space
        - b: number of bootstrap samples
        - alpha: IC test on outcome model
        output:
        - coef: calculated using bootstrap
        Note: Due to time constrains, only one PPCA is fitted
        """
        x, x_val, holdout_mask = self.daHoldout(holdout_prop)
        print('... Done Holdout')
        w, z, x_gen = self.FM_Prob_PCA(x, True)
        print('... Done PPCA')
        pvalue = self.PredictiveCheck(x_val, x_gen, w, z, holdout_mask)
        low = stats.norm(0, 1).ppf(alpha / 2)
        up = stats.norm(0, 1).ppf(1 - alpha / 2)
        del x_gen
        if 0.1 < pvalue < 0.9:
            print('... Pass Predictive Check: ' + str(pvalue))
            print('... Fitting Outcome Model')
            coef = []
            pca = np.transpose(z)

            # Bootstrap to calculate the coefs
            for i in range(b):
                rows = np.random.choice(self.X_train.shape[0], int(self.X_train.shape[0] * 0.85), replace=False)
                coef_, _ = self.OutcomeModel_LR(pca, rows, roc_flag=False)
                coef.append(coef_)
            coef = np.matrix(coef)
            coef = coef[:, 0:self.X_train.shape[1]]  # ?

            # Building IC
            coef_m = np.asarray(np.mean(coef, axis=0)).reshape(-1)
            coef_var = np.asarray(np.var(coef, axis=0)).reshape(-1)
            coef_z = np.divide(coef_m, np.sqrt(coef_var / b))
            coef_z = [1 if low < c < up else 0 for c in coef_z]  # 1 if significative and 0 otherwise

            # if ROC = TRUE, calculate ROC results and score is just for testing set
            del coef_var, coef, coef_
            # w, z, x_gen = FM_Prob_PCA(train, k, False)
            _, roc = self.OutcomeModel_LR(pca, roc_flag=True)
        else:
            print('Failed on Predictive Check. Suggetions: trying a different K')
            coef_m = []
            coef_z = []
            roc = []

        return np.multiply(coef_m, coef_z), coef_m, roc

    def daHoldout(self, holdout_portion):
        """
        Hold out a few values from train set to calculate predictive check
        """
        n_holdout = int(holdout_portion * self.n * self.ncol)
        holdout_row = np.random.randint(self.n, size=n_holdout)
        holdout_col = np.random.randint(self.ncol, size=n_holdout)
        holdout_mask = (
            sparse.coo_matrix((np.ones(n_holdout), (holdout_row, holdout_col)), shape=self.X.shape)).toarray()
        # holdout_subjects = np.unique(holdout_row)
        holdout_mask = np.minimum(1, holdout_mask)
        x_train = np.multiply(1 - holdout_mask, self.X)
        x_val = np.multiply(holdout_mask, self.X)
        return x_train, x_val, holdout_mask

    def FM_Prob_PCA(self, x, flag_pred=False, stddv_datapoints=1):
        """
        Factor Model: Probabilistic PCA
        input:
            x: data
            k: size of latent variables
            flag_pred: if values to calculate the predictive check should be saved
        output:
            w and z values, generated sample to predictive check
        """
        # Reference: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_PCA.ipynb
        from tensorflow.keras import optimizers
        import tensorflow as tf
        import tensorflow_probability as tfp
        from tensorflow_probability import distributions as tfd
        # tf.enable_eager_execution()

        def PPCA(stddv_datapoints):
            """
            Calculating sub parts of PPCA
            """
            w = yield Root(tfd.Independent(
                tfd.Normal(loc=tf.zeros([self.n, self.k]),
                           scale=2.0 * tf.ones([self.n, self.k]),
                           name="w"), reinterpreted_batch_ndims=2))
            z = yield Root(tfd.Independent(
                tfd.Normal(loc=tf.zeros([self.k, self.ncol]),
                           scale=tf.ones([self.k, self.ncol]),
                           name="z"), reinterpreted_batch_ndims=2))
            x = yield tfd.Independent(tfd.Normal(
                loc=tf.matmul(w, z),
                scale=stddv_datapoints,
                name="x"), reinterpreted_batch_ndims=2)

        def factored_normal_variational_model():
            qw = yield Root(tfd.Independent(tfd.Normal(
                loc=qw_mean, scale=qw_stddv, name="qw"), reinterpreted_batch_ndims=2))
            qz = yield Root(tfd.Independent(tfd.Normal(
                loc=qz_mean, scale=qz_stddv, name="qz"), reinterpreted_batch_ndims=2))

        x_train = tf.convert_to_tensor(x, dtype=tf.float32)
        # x_train = tf.convert_to_tensor(np.transpose(x), dtype=tf.float32)
        Root = tfd.JointDistributionCoroutine.Root
        # num_datapoints, data_dim = x.shape
        # data_dim, num_datapoints = x_train.shape
        concrete_ppca_model = functools.partial(PPCA, stddv_datapoints=stddv_datapoints)
        model = tfd.JointDistributionCoroutine(concrete_ppca_model)
        w = tf.Variable(np.ones([self.n, self.k]), dtype=tf.float32)
        z = tf.Variable(np.ones([self.k, self.ncol]), dtype=tf.float32)

        target_log_prob_fn = lambda w, z: model.log_prob((w, z, x_train))
        losses = tfp.math.minimize(lambda: -target_log_prob_fn(w, z),
                                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                                   num_steps=200)
        qw_mean = tf.Variable(np.ones([self.n, self.k]), dtype=tf.float32)
        qz_mean = tf.Variable(np.ones([self.k, self.ncol]), dtype=tf.float32)
        qw_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.n, self.k]), dtype=tf.float32))
        qz_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.k, self.ncol]), dtype=tf.float32))

        surrogate_posterior = tfd.JointDistributionCoroutine(factored_normal_variational_model)

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
        return w.numpy(), z.numpy(), x_generated

    def PredictiveCheck(self, x_val, x_gen, w, z, holdout_mask):
        """
        calculate the predictive check
        input:
            x_val: observed values
            x_gen: generated values
            w, z: from fm_PPCA
            holdout_mask
        output:
            pvalue from the predictive check
        """
        # Data prep, holdout mask operations
        holdout_mask1 = np.asarray(holdout_mask).reshape(-1)
        x_val1 = np.asarray(x_val).reshape(-1)
        x1 = np.asarray(np.multiply(np.dot(w, z), holdout_mask)).reshape(-1)
        del x_val
        x_val1 = x_val1[holdout_mask1 == 1]
        x1 = x1[holdout_mask1 == 1]
        pvals = np.zeros(len(x_gen))
        for i in range(len(x_gen)):
            holdout_sample = np.multiply(x_gen[i], holdout_mask)
            holdout_sample = np.asarray(holdout_sample).reshape(-1)
            holdout_sample = holdout_sample[holdout_mask1 == 1]
            x_val_current = stats.norm(holdout_sample, 1).logpdf(x_val1)
            x_gen_current = stats.norm(holdout_sample, 1).logpdf(x1)
            pvals[i] = np.mean(np.array(x_val_current < x_gen_current))
        return np.mean(pvals)

    def OutcomeModel_LR(self, pca, rows=None, roc_flag=True):
        """
        outcome model from the DA
        input:
        - x: training set
        - x_latent: output from factor model
        - y01: outcome
        """
        import scipy.stats as st
        model = linear_model.SGDClassifier(penalty='l2', alpha=0.1, l1_ratio=0.01, loss='modified_huber',
                                           fit_intercept=True, random_state=0)
        if roc_flag:
            rows_train = range(self.X_train.shape[0])
            rows_test = range(self.X_train.shape[0] + 1, self.X_train.shape[0] + self.X_test.shape[0] + 1)
            assert len(rows_train) == len(self.y_train), "Error training set dimensions"
            assert len(rows_test) == len(self.y_test), "Error testing set dimensions"

            X_train = np.concatenate([self.X_train, pca[rows_train, :]], axis=1)
            X_test = np.concatenate([self.X_test, pca[rows_test, :]], axis=1)

            modelcv = calibration.CalibratedClassifierCV(base_estimator=model,
                                                         cv=5, method='isotonic').fit(X_train, self.y_train)
            coef = []
            y_test_pred = modelcv.predict(X_test)
            y_test_predp = modelcv.predict_proba(X_test)
            y_train_pred = modelcv.predict(X_train)

            y_test_predp1 = [i[1] for i in y_test_predp]
            print('... Leaner Evaluation:')

            print('... Training set: F1 - ', f1_score(self.y_train, y_train_pred),
                  sum(y_train_pred), sum(self.y_train))
            print('...... confusion matrix: \n', confusion_matrix(self.y_train, y_train_pred))

            print('... Testing set: F1 - ', f1_score(self.y_test, y_test_pred), sum(y_test_pred), sum(self.y_test))
            print('...... confusion matrix: \n', confusion_matrix(self.y_test, y_test_pred))
            fpr, tpr, _ = roc_curve(self.y_test, y_test_predp1)
            auc = roc_auc_score(self.y_test, y_test_predp1)
            roc = {'learners': 'DA',
                   'fpr': fpr,
                   'tpr': tpr,
                   'auc': auc}
        else:
            x_aug = np.concatenate([self.X_train[rows, :], pca[rows, :]], axis=1)
            model.fit(x_aug, self.y_train[rows])
            coef = model.coef_[0]
            roc = {}
        return coef, roc