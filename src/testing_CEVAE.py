# -*- coding: utf-8 -*-
"""
Reference: https://github.com/AMLab-Amsterdam/CEVAE
"""

import numpy as np
from sklearn.model_selection import train_test_split

#datasets.py
class IHDP(object):
    def __init__(self, path_data="datasets/IHDP/csv", replications=10):
        self.path_data = path_data
        self.replications = replications
        # which features are binary
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
        self.contfeats = [i for i in range(25) if i not in self.binfeats]

    def __iter__(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            # this binary feature is in {1, 2}
            x[:, 13] -= 1
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, self.contfeats, self.binfeats


class Evaluator(object):
    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.mu0 = mu0
        self.mu1 = mu1
        if mu0 is not None and mu1 is not None:
            self.true_ite = mu1 - mu0

    def rmse_ite(self, ypred1, ypred0):
        pred_ite = np.zeros_like(self.true_ite)
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return np.sqrt(np.mean(np.square(self.true_ite - pred_ite)))

    def abs_ate(self, ypred1, ypred0):
        return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite))

    def pehe(self, ypred1, ypred0):
        return np.sqrt(np.mean(np.square((self.mu1 - self.mu0) - (ypred1 - ypred0))))

    def y_errors(self, y0, y1):
        ypred = (1 - self.t) * y0 + self.t * y1
        ypred_cf = self.t * y0 + (1 - self.t) * y1
        return self.y_errors_pcf(ypred, ypred_cf)

    def y_errors_pcf(self, ypred, ypred_cf):
        rmse_factual = np.sqrt(np.mean(np.square(ypred - self.y)))
        rmse_cfactual = np.sqrt(np.mean(np.square(ypred_cf - self.y_cf)))
        return rmse_factual, rmse_cfactual

    def calc_stats(self, ypred1, ypred0):
        ite = self.rmse_ite(ypred1, ypred0)
        ate = self.abs_ate(ypred1, ypred0)
        pehe = self.pehe(ypred1, ypred0)
        return ite, ate, pehe

#### MAIN MODEL


import edward2 as ed
import tensorflow as tf
import tensorflow_probability as tfp

from edward2 import Bernoulli, Normal
from progressbar import ETA, Bar, Percentage, ProgressBar

#from datasets import IHDP
#from evaluation import Evaluator
import numpy as np
import time
from scipy.stats import sem
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import initializers

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-reps', type=int, default=10)
parser.add_argument('-earl', type=int, default=10)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-opt', choices=['adam', 'adamax'], default='adam')
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-print_every', type=int, default=10)
args = parser.parse_args()

args.true_post = True

#using train from real-world application 
#first column treatment, second column y and others covariates

#dataset = IHDP(replications=args.reps)
dataset = train
dimx = 25
scores = np.zeros((args.reps, 3))
scores_test = np.zeros((args.reps, 3))

M = None  # batch size during training
d = 20  # latent dimension
lamba = 1e-4  # weight decay
nh, h = 3, 200  # number and size of hidden layers

for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
    print('\nReplication {}/{}'.format(i + 1, args.reps))
    (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
    (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
    (xte, tte, yte), (y_cfte, mu0te, mu1te) = test
    evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)

    # reorder features with binary first and continuous after
    perm = binfeats + contfeats
    xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]

    xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), np.concatenate([ttr, tva], axis=0), np.concatenate([ytr, yva], axis=0)
    evaluator_train = Evaluator(yalltr, talltr, y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
                                mu0=np.concatenate([mu0tr, mu0va], axis=0), mu1=np.concatenate([mu1tr, mu1va], axis=0))

    # zero mean, unit variance for y during training
    ym, ys = np.mean(ytr), np.std(ytr)
    ytr, yva = (ytr - ym) / ys, (yva - ym) / ys
    best_logpvalid = - np.inf

    with tf.Graph().as_default():
        sess = tf.InteractiveSession()

        ed.set_seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)

        x_ph_bin = tf.placeholder(tf.float32, [M, len(binfeats)], name='x_bin')  # binary inputs
        x_ph_cont = tf.placeholder(tf.float32, [M, len(contfeats)], name='x_cont')  # continuous inputs
        t_ph = tf.placeholder(tf.float32, [M, 1])
        y_ph = tf.placeholder(tf.float32, [M, 1])

        x_ph = tf.concat([x_ph_bin, x_ph_cont], 1)
        activation = tf.nn.elu

        # CEVAE model (decoder)
        # p(z)
        z = Normal(loc=tf.zeros([tf.shape(x_ph)[0], d]), scale=tf.ones([tf.shape(x_ph)[0], d]))

        # p(x|z)
        hx = fc_net(z, (nh - 1) * [h], [], 'px_z_shared', lamba=lamba, activation=activation)
        logits = fc_net(hx, [h], [[len(binfeats), None]], 'px_z_bin'.format(i + 1), lamba=lamba, activation=activation)
        x1 = Bernoulli(logits=logits, dtype=tf.float32, name='bernoulli_px_z')

        mu, sigma = fc_net(hx, [h], [[len(contfeats), None], [len(contfeats), tf.nn.softplus]], 'px_z_cont', lamba=lamba,
                           activation=activation)
        x2 = Normal(loc=mu, scale=sigma, name='gaussian_px_z')

        # p(t|z)
        logits = fc_net(z, [h], [[1, None]], 'pt_z', lamba=lamba, activation=activation)
        t = Bernoulli(logits=logits, dtype=tf.float32)

        # p(y|t,z)
        mu2_t0 = fc_net(z, nh * [h], [[1, None]], 'py_t0z', lamba=lamba, activation=activation)
        mu2_t1 = fc_net(z, nh * [h], [[1, None]], 'py_t1z', lamba=lamba, activation=activation)
        y = Normal(loc=t * mu2_t1 + (1. - t) * mu2_t0, scale=tf.ones_like(mu2_t0))

        # CEVAE variational approximation (encoder)
        # q(t|x)
        logits_t = fc_net(x_ph, [d], [[1, None]], 'qt', lamba=lamba, activation=activation)
        qt = Bernoulli(logits=logits_t, dtype=tf.float32)
        # q(y|x,t)
        hqy = fc_net(x_ph, (nh - 1) * [h], [], 'qy_xt_shared', lamba=lamba, activation=activation)
        mu_qy_t0 = fc_net(hqy, [h], [[1, None]], 'qy_xt0', lamba=lamba, activation=activation)
        mu_qy_t1 = fc_net(hqy, [h], [[1, None]], 'qy_xt1', lamba=lamba, activation=activation)
        qy = Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))
        # q(z|x,t,y)
        inpt2 = tf.concat([x_ph, qy], 1)
        hqz = fc_net(inpt2, (nh - 1) * [h], [], 'qz_xty_shared', lamba=lamba, activation=activation)
        muq_t0, sigmaq_t0 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt0', lamba=lamba,
                                   activation=activation)
        muq_t1, sigmaq_t1 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt1', lamba=lamba,
                                   activation=activation)
        qz = Normal(loc=qt * muq_t1 + (1. - qt) * muq_t0, scale=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

        # Create data dictionary for edward
        data = {x1: x_ph_bin, x2: x_ph_cont, y: y_ph, qt: t_ph, t: t_ph, qy: y_ph}

        # sample posterior predictive for p(y|z,t)
        y_post = ed.copy(y, {z: qz, t: t_ph}, scope='y_post')
        # crude approximation of the above
        y_post_mean = ed.copy(y, {z: qz.mean(), t: t_ph}, scope='y_post_mean')
        # construct a deterministic version (i.e. use the mean of the approximate posterior) of the lower bound
        # for early stopping according to a validation set
        y_post_eval = ed.copy(y, {z: qz.mean(), qt: t_ph, qy: y_ph, t: t_ph}, scope='y_post_eval')
        x1_post_eval = ed.copy(x1, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='x1_post_eval')
        x2_post_eval = ed.copy(x2, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='x2_post_eval')
        t_post_eval = ed.copy(t, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='t_post_eval')
        logp_valid = tf.reduce_mean(tf.reduce_sum(y_post_eval.log_prob(y_ph) + t_post_eval.log_prob(t_ph), axis=1) +
                                    tf.reduce_sum(x1_post_eval.log_prob(x_ph_bin), axis=1) +
                                    tf.reduce_sum(x2_post_eval.log_prob(x_ph_cont), axis=1) +
                                    tf.reduce_sum(z.log_prob(qz.mean()) - qz.log_prob(qz.mean()), axis=1))

        inference = ed.KLqp({z: qz}, data)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        inference.initialize(optimizer=optimizer)

        saver = tf.train.Saver(tf.contrib.slim.get_variables())
        tf.global_variables_initializer().run()

        n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int(xtr.shape[0] / 100), np.arange(xtr.shape[0])

        # dictionaries needed for evaluation
        tr0, tr1 = np.zeros((xalltr.shape[0], 1)), np.ones((xalltr.shape[0], 1))
        tr0t, tr1t = np.zeros((xte.shape[0], 1)), np.ones((xte.shape[0], 1))
        f1 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):], t_ph: tr1}
        f0 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):], t_ph: tr0}
        f1t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):], t_ph: tr1t}
        f0t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):], t_ph: tr0t}

        for epoch in range(n_epoch):
            avg_loss = 0.0

            t0 = time.time()
            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(n_iter_per_epoch, widgets=widgets)
            pbar.start()
            np.random.shuffle(idx)
            for j in range(n_iter_per_epoch):
                pbar.update(j)
                batch = np.random.choice(idx, 100)
                x_train, y_train, t_train = xtr[batch], ytr[batch], ttr[batch]
                info_dict = inference.update(feed_dict={x_ph_bin: x_train[:, 0:len(binfeats)],
                                                        x_ph_cont: x_train[:, len(binfeats):],
                                                        t_ph: t_train, y_ph: y_train})
                avg_loss += info_dict['loss']

            avg_loss = avg_loss / n_iter_per_epoch
            avg_loss = avg_loss / 100

            if epoch % args.earl == 0 or epoch == (n_epoch - 1):
                logpvalid = sess.run(logp_valid, feed_dict={x_ph_bin: xva[:, 0:len(binfeats)], x_ph_cont: xva[:, len(binfeats):],
                                                            t_ph: tva, y_ph: yva})
                if logpvalid >= best_logpvalid:
                    print('Improved validation bound, old: {:0.3f}, new: {:0.3f}'.format(best_logpvalid, logpvalid))
                    best_logpvalid = logpvalid
                    saver.save(sess, 'models/m6-ihdp')

            if epoch % args.print_every == 0:
                y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=1)
                y0, y1 = y0 * ys + ym, y1 * ys + ym
                score_train = evaluator_train.calc_stats(y1, y0)
                rmses_train = evaluator_train.y_errors(y0, y1)

                y0, y1 = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=1)
                y0, y1 = y0 * ys + ym, y1 * ys + ym
                score_test = evaluator_test.calc_stats(y1, y0)

                print("Epoch: {}/{}, log p(x) >= {:0.3f}, ite_tr: {:0.3f}, ate_tr: {:0.3f}, pehe_tr: {:0.3f}, " \
                      "rmse_f_tr: {:0.3f}, rmse_cf_tr: {:0.3f}, ite_te: {:0.3f}, ate_te: {:0.3f}, pehe_te: {:0.3f}, " \
                      "dt: {:0.3f}".format(epoch + 1, n_epoch, avg_loss, score_train[0], score_train[1], score_train[2],
                                           rmses_train[0], rmses_train[1], score_test[0], score_test[1], score_test[2],
                                           time.time() - t0))

        saver.restore(sess, 'models/m6-ihdp')
        y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=100)
        y0, y1 = y0 * ys + ym, y1 * ys + ym
        score = evaluator_train.calc_stats(y1, y0)
        scores[i, :] = score

        y0t, y1t = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=100)
        y0t, y1t = y0t * ys + ym, y1t * ys + ym
        score_test = evaluator_test.calc_stats(y1t, y0t)
        scores_test[i, :] = score_test

        print('Replication: {}/{}, tr_ite: {:0.3f}, tr_ate: {:0.3f}, tr_pehe: {:0.3f}' \
              ', te_ite: {:0.3f}, te_ate: {:0.3f}, te_pehe: {:0.3f}'.format(i + 1, args.reps,
                                                                            score[0], score[1], score[2],
                                                                            score_test[0], score_test[1], score_test[2]))
        sess.close()

print('CEVAE model total scores')
means, stds = np.mean(scores, axis=0), sem(scores, axis=0)
print('train ITE: {:.3f}+-{:.3f}, train ATE: {:.3f}+-{:.3f}, train PEHE: {:.3f}+-{:.3f}' \
      ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))

means, stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
print('test ITE: {:.3f}+-{:.3f}, test ATE: {:.3f}+-{:.3f}, test PEHE: {:.3f}+-{:.3f}' \
      ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))
    
    
    
#utils
def fc_net(inp, layers, out_layers, scope, lamba=1e-3, activation=tf.nn.relu, reuse=None,
           weights_initializer=initializers.xavier_initializer(uniform=False)):
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=activation,
                        normalizer_fn=None,
                        weights_initializer=weights_initializer,
                        reuse=reuse,
                        weights_regularizer=slim.l2_regularizer(lamba)):

        if layers:
            h = slim.stack(inp, slim.fully_connected, layers, scope=scope)
            if not out_layers:
                return h
        else:
            h = inp
        outputs = []
        for i, (outdim, activation) in enumerate(out_layers):
            o1 = slim.fully_connected(h, outdim, activation_fn=activation, scope=scope + '_{}'.format(i + 1))
            outputs.append(o1)
        return outputs if len(outputs) > 1 else outputs[0]


def get_y0_y1(sess, y, f0, f1, shape=(), L=1, verbose=True):
    y0, y1 = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    ymean = y.mean()
    for l in range(L):
        if L > 1 and verbose:
            sys.stdout.write('\r Sample {}/{}'.format(l + 1, L))
            sys.stdout.flush()
        y0 += sess.run(ymean, feed_dict=f0) / L
        y1 += sess.run(ymean, feed_dict=f1) / L

    if L > 1 and verbose:
        print()
    return y0, y1
