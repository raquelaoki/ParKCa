#Reference https://github.com/kim-hyunsu/CEVAE-pyro/blob/master/model/vae.py

#main
from sklearn.preprocessing import MinMaxScaler

from argparse import ArgumentParser
from CEVAE_initialisation import init_qz
from CEVAE_dataset import DATA
from CEVAE_evaluation import Evaluator, get_y0_y1
from CEVAE_networks import p_x_z, p_t_z, p_y_zt, q_t_x, q_y_xt, q_z_tyx

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch.distributions import normal
from torch import optim
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

# set random seeds:
# torch.manual_seed(7)
# np.random.seed(7)
parser = ArgumentParser()
# Set Hyperparameters
parser.add_argument('-reps', type=int, default=10000)
parser.add_argument('-z_dim', type=int, default=20)
parser.add_argument('-h_dim', type=int, default=64)
parser.add_argument('-epochs', type=int, default=3 ) #change to 100
parser.add_argument('-batch', type=int, default=500)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-decay', type=float, default=0.001)
parser.add_argument('-print_every', type=int, default=10)

args = parser.parse_args()

def model(n_causes,  data_path, y01):

    dataset = DATA(ncol=n_causes-1,data_path = data_path,y01 = y01)


    # Loop for replications
    for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):

        #i, (train, valid, test, contfeats, binfeats) = dataset.get_train_valid_test()
        print('\nReplication %i/%i' % (i + 1, args.reps))
        # read out data
        (xtr, ttr, ytr) = train
        (xva, tva, yva) = valid
        (xte, tte, yte) = test

        # concatenate train and valid for training
        xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), np.concatenate([ttr, tva], axis=0), np.concatenate(
            [ytr, yva], axis=0)

        try:
            # set evaluator objects
            evaluator_train = Evaluator(yalltr, talltr )
            evaluator_test = Evaluator(yte, tte)

            # zero mean, unit variance for y during training, use ym & ys to correct when using testset
            ym, ys = np.mean(ytr), np.std(ytr)
            ytr, yva = (ytr - ym) / ys, (yva - ym) / ys

            # init networks (overwritten per replication)
            x_dim = len(binfeats) + len(contfeats)
            p_x_z_dist = p_x_z(dim_in=args.z_dim, nh=3, dim_h=args.h_dim, dim_out_bin=len(binfeats),
                               dim_out_con=len(contfeats)).cuda()
            p_t_z_dist = p_t_z(dim_in=args.z_dim, nh=1, dim_h=args.h_dim, dim_out=1).cuda()
            p_y_zt_dist = p_y_zt(dim_in=args.z_dim, nh=3, dim_h=args.h_dim, dim_out=1).cuda()
            q_t_x_dist = q_t_x(dim_in=x_dim, nh=1, dim_h=args.h_dim, dim_out=1).cuda()
            # t is not feed into network, therefore not increasing input size (y is fed).
            q_y_xt_dist = q_y_xt(dim_in=x_dim, nh=3, dim_h=args.h_dim, dim_out=1).cuda()
            q_z_tyx_dist = q_z_tyx(dim_in=len(binfeats) + len(contfeats) +1, nh=3, dim_h=args.h_dim,
                                   dim_out=args.z_dim).cuda() #remove an 1 from dim_in
            p_z_dist = normal.Normal(torch.zeros(args.z_dim).cuda(), torch.ones(args.z_dim).cuda())

            # Create optimizer
            params = list(p_x_z_dist.parameters()) + \
                     list(p_t_z_dist.parameters()) + \
                     list(p_y_zt_dist.parameters()) + \
                     list(q_t_x_dist.parameters()) + \
                     list(q_y_xt_dist.parameters()) + \
                     list(q_z_tyx_dist.parameters())

            # Adam is used, like original implementation, in paper Adamax is suggested
            optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.decay)

            # init q_z inference
            q_z_tyx_dist = init_qz(q_z_tyx_dist, p_z_dist, ytr, ttr, xtr)

            # set batch size
            M = args.batch

            n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int(xtr.shape[0] / M), list(range(xtr.shape[0]))

            loss = defaultdict(list)
            for epoch in range(n_epoch):
                # print('Epoch: %i/%i' % (epoch, n_epoch))
                loss_sum = 0.
                # shuffle index
                np.random.shuffle(idx)
                # take random batch for training
                for j in range(n_iter_per_epoch):
                    # select random batch
                    batch = np.random.choice(idx, M)
                    x_train, y_train, t_train = torch.cuda.FloatTensor(xalltr[batch]), torch.cuda.FloatTensor(yalltr[batch]), \
                                                torch.cuda.FloatTensor(talltr[batch])

                    # inferred distribution over z
                    xy = torch.cat((x_train, y_train), 1)
                    z_infer = q_z_tyx_dist(xy=xy, t=t_train)
                    # use a single sample to approximate expectation in lowerbound
                    z_infer_sample = z_infer.sample()

                    # RECONSTRUCTION LOSS
                    # p(x|z)
                    x_bin, x_con = p_x_z_dist(z_infer_sample)
                    l1 = x_bin.log_prob(x_train[:, :len(binfeats)]).sum(1)
                    loss['Reconstr_x_bin'].append(l1.sum().cpu().detach().float())
                    #l2 = x_con.log_prob(x_train[:, -len(contfeats):]).sum(1)
                    #loss['Reconstr_x_con'].append(l2.sum().cpu().detach().float())
                    # p(t|z)
                    t = p_t_z_dist(z_infer_sample)
                    l3 = t.log_prob(t_train).squeeze()
                    loss['Reconstr_t'].append(l3.sum().cpu().detach().float())
                    # p(y|t,z)
                    # for training use t_train, in out-of-sample prediction this becomes t_infer
                    y = p_y_zt_dist(z_infer_sample, t_train)
                    l4 = y.log_prob(y_train).squeeze()
                    loss['Reconstr_y'].append(l4.sum().cpu().detach().float())

                    # REGULARIZATION LOSS
                    # p(z) - q(z|x,t,y)
                    # approximate KL
                    l5 = (p_z_dist.log_prob(z_infer_sample) - z_infer.log_prob(z_infer_sample)).sum(1)
                    # Analytic KL (seems to make overall performance less stable)
                    # l5 = (-torch.log(z_infer.stddev) + 1/2*(z_infer.variance + z_infer.mean**2 - 1)).sum(1)
                    loss['Regularization'].append(l5.sum().cpu().detach().float())

                    # AUXILIARY LOSS
                    # q(t|x)
                    t_infer = q_t_x_dist(x_train)
                    l6 = t_infer.log_prob(t_train).squeeze()
                    loss['Auxiliary_t'].append(l6.sum().cpu().detach().float())
                    # q(y|x,t)
                    y_infer = q_y_xt_dist(x_train, t_train)
                    l7 = y_infer.log_prob(y_train).squeeze()
                    loss['Auxiliary_y'].append(l7.sum().cpu().detach().float())

                    # Total objective
                    # inner sum to calculate loss per item, torch.mean over batch
                    loss_mean = torch.mean(l1 + l3 + l4 + l5 + l6 + l7) #+ l2
                    loss['Total'].append(loss_mean.cpu().detach().numpy())
                    objective = -loss_mean

                    optimizer.zero_grad()
                    # Calculate gradients
                    objective.backward()
                    # Update step
                    optimizer.step()

                if epoch % args.print_every == 0:
                    print('Epoch %i' % epoch)
                    #y0, y1 = get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, torch.tensor(xalltr).cuda(),
                    #                   torch.tensor(talltr).cuda())

                    #testing set
                    y0, y1 = get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, torch.tensor(xte).cuda(),
                                       torch.tensor(tte).cuda())
                    y0, y1 = y0 * ys + ym, y1 * ys + ym



            y0, y1 = get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, torch.tensor(xte).cuda(),
                               torch.tensor(tte).cuda())
            y0, y1 = y0 * ys + ym, y1 * ys + ym

            y01_pred = q_y_xt_dist( torch.cuda.FloatTensor(xte), torch.cuda.FloatTensor(tte))
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(y01_pred.mean.cpu().detach().numpy())
            y_pred = scaler.transform(y01_pred.mean.cpu().detach().numpy())

            fig = plt.figure(figsize=(6,4))
            plt.plot(loss['Total'], label='Total')
            plt.title('Variational Lower Bound',fontsize=15)
            plt.show()
            fig.savefig('results//plots_cevae//cevae_sim0_t'+str(i)+'.png')

            yield (y0[:,0].mean(), y1[:,0].mean(),(y1[:,0]-y0[:,0]).mean() ,y_pred,  np.squeeze(np.asarray(yte)))
        except ValueError:
            y_pred = np.empty(len(yte))
            y_pred[:]=np.Nan
            yield (0.0, 0.0,0.0,y_pred, np.squeeze(np.asarray(yte)))
