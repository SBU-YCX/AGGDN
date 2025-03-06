"""
Author: Yingru Liu, Yucheng Xing
"""

import time
import logging 
import argparse, os, copy
import torch
import torch.optim as optim
import scipy.io as sio
import numpy as np
from tqdm.auto import tqdm
from processing import RTDS_data
from torch.utils.data import DataLoader
from models import NGDCSDE, Discriminator

# RTDS: 32

#########################################################################################################################
parser = argparse.ArgumentParser(description='Experiment Configuration.')
# Training setting.
parser.add_argument('--dataset',
                    required=False,
                    default='rtds_nodes',
                    type=str,
                    help='dataset.')
parser.add_argument('--nf',
                    required=False,
                    default=100,
                    type=int,
                    help='length of sequence.')
parser.add_argument('--dr',
                    required=False,
                    default=1,
                    type=int,
                    help='data sampling rate')
parser.add_argument('--p',
                    required=False,
                    default=0.8,
                    type=float,
                    help='ratio of observability.')
parser.add_argument('--np',
                    required=False,
                    default=0.1,
                    type=float,
                    help='ratio of noise.')
parser.add_argument('--p_spatial',
                    required=False,
                    default=0.6,
                    type=float,
                    help='spatial ratio of observability.')
parser.add_argument('--p_temporal',
                    required=False,
                    default=0.4,
                    type=float,
                    help='temporal ratio of observability.')
parser.add_argument('--downsampling',
                    required=False,
                    default=1,
                    type=int,
                    help='downsampling ratio in time axis.')
parser.add_argument('--data_path',
                    required=False,
                    default='datasets',
                    type=str,
                    help='path that saves the data file (w.o filename)')
parser.add_argument('--return_y',
                    dest='return_y',
                    action='store_true',
                    default=False)
parser.add_argument('--batch_size',
                    required=False,
                    default=200,
                    type=int,
                    help='batch size.')
parser.add_argument('--max_epoch',
                    required=False,
                    default=100,
                    type=int,
                    help='max epoch.')
parser.add_argument('--lr',
                    required=False,
                    default=1e-3,
                    type=float,
                    help='learning rate.')
parser.add_argument('--model_name',
                    required=False,
                    default="",
                    type=str,
                    help='type of model.')
parser.add_argument('--saveto',
                    required=False,
                    default="./results_latest",
                    type=str,
                    help='path to save model.')
parser.add_argument('--checkname',
                    required=False,
                    default=None,
                    type=str,
                    help='path to load model.')
parser.add_argument('--early_stop',
                    required=False,
                    default=10,
                    type=int,
                    help='epoch tolerance for early stopping .')
parser.add_argument('--learnstd',
                    dest='learnstd',
                    action='store_true',
                    default=False)
# Common Model Setting.
parser.add_argument('--dimIn',
                    required=False,
                    default=2,
                    type=int,
                    help='the dimension of node input.')
parser.add_argument('--numNode',
                    required=False,
                    default=33,
                    type=int,
                    help='the number of nodes in the graph.')
parser.add_argument('--delta_t',
                    required=False,
                    default=0.01,
                    type=float,
                    help='the time step for continuous-time model.')
# STGCN Setting.
parser.add_argument('--numBlock',
                    required=False,
                    default=2,
                    type=int,
                    help='number of STGCN blocks.')
parser.add_argument(
    '--dimOutConv1',
    required=False,
    default=32,
    type=int,
    help=
    'output dimension of the first temporal convolutional layer (NODE) of STGCN block.'
)
parser.add_argument(
    '--dimGCN',
    required=False,
    default=32,
    type=int,
    help='output dimension of graph convolutional layers of STGCN block.')
parser.add_argument(
    '--dimOutConv2',
    required=False,
    default=32,
    type=int,
    help=
    'output dimension of the second temporal convolutional layer (NODE) of STGCN block.'
)
parser.add_argument(
    '--kernel_size',
    required=False,
    default=12,
    type=int,
    help=
    'kernel size of the first temporal convolutional layers of STGCN block.')
# .
parser.add_argument('--dimHidden',
                    required=False,
                    default=32,
                    type=int,
                    help='output dimension of the EGCU-H (Node).')
# GRUODE&ODERNN&GCGRU Setting.
parser.add_argument('--dimRnn',
                    required=False,
                    default=32,
                    type=int,
                    help='output dimension of GRUCell (Node).')
parser.add_argument('--numRNNs',
                    required=False,
                    default=1,
                    type=int,
                    help='number of RNN blocks.')
# DCGRU Setting.
parser.add_argument('--numRamWalks',
                    required=False,
                    default=3,
                    type=int,
                    help='number of random walks of diffusion convolution.')
# ODERNN Setting.
parser.add_argument('--dimODEHidden',
                    required=False,
                    default=32,
                    type=int,
                    help='hidden dimension of the ODE network.')
parser.add_argument('--numODEHidden',
                    required=False,
                    default=1,
                    type=int,
                    help='number of hidden layer of the ODE network.')
parser.add_argument('--dimState',
                    required=False,
                    default=3,
                    type=int,
                    help='latent state dimension of SDE.')
parser.add_argument('--alpha',
                    required=False,
                    default=1.0,
                    type=float,
                    help='coefficient of adversarial loss.')
parser.add_argument('--beta',
                    required=False,
                    default=1.0,
                    type=float,
                    help='coefficient of adversarial loss.')
parser.add_argument('--numTraj',
                    required=False,
                    default=50,
                    type=int,
                    help='number of trajectories of latent state.')
parser.add_argument('--odeMethod', 
                    required = False, 
                    default = 'euler',
                    type = str,
                    help = 'method of ode solver used.'
                    )
parser.add_argument('--sdeMethod', 
                    required = False, 
                    default = 'euler',
                    type = str,
                    help = 'method of sde solver used.'
                    )
parser.add_argument('--sdeConnect', 
                    required = False, 
                    default = 'r',
                    type = str,
                    help = 'structure of sde'
                    )
parser.add_argument('--useGru', 
                    required = False, 
                    default = True, 
                    type = bool, 
                    help = 'use gru or not')
#
#########################################################################################################################


def get_logger(log_file): 
    logging.basicConfig(level = logging.DEBUG)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if log_file: 
        file_handler = logging.FileHandler(log_file)
        logger.addHandler(file_handler)
    return logger


def get_ckp_path(args):
    main_path = args.saveto
    sub_folder = '{}_spatial{}_temporal{}_downsampling{}'.format(args.dataset, int(args.p_spatial // 0.0999), int(args.p_temporal // 0.0999), args.downsampling)
    if args.model_name.lower() == 'ngdcsde':
        file_name = '{}_{}_vRnn{}_vLatent{}_#numRNNs{}_deltaT{}_lambda{}_#randomWalk{}_#traj{}_{}.pth'.format(
            args.model_name, args.sdeConnect, args.dimRnn, args.dimState, args.numRNNs,
            args.delta_t, int(args.beta // 0.00999), args.numRamWalks, args.numTraj, args.adversarial)
    else:
        file_name = 'unknown_model.pth'
    discriminator_name = '{}_discriminator.pth'.format(args.model_name)
    sub_path = os.path.join(main_path, sub_folder)
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
    return os.path.join(sub_path, file_name), sub_path, os.path.join(sub_path, discriminator_name)


def get_model(args):
    if args.model_name.lower() == 'ngdcsde':
        return NGDCSDE(args.dimIn,
                       args.dimRnn,
                       args.dimState,
                       args.numRNNs,
                       args.dimODEHidden,
                       args.numODEHidden,
                       args.numNode, 
                       args.delta_t,
                       args.beta,
                       args.numTraj,
                       args.numRamWalks,
                       args.learnstd).cuda()
    return


def get_dataset(args):
    if args.dataset.lower() == 'rtds_noisy_node': 
        filename = '{}_len{}_missing_points_spatial{}_temporal{}_downsampling{}.hdf5'.format(
            'rtds_noisy', args.nf, int(args.p_spatial // 0.0999),
            int(args.p_temporal // 0.0999), args.downsampling)
        data_path = os.path.join(args.data_path, filename)
        data_train = RTDS_data(split='train',
                               mask_path=data_path,
                               return_Y=args.return_y,
                               return_Edge=False,
                               delta_t=0.01, 
                               noise_ratio=args.np, 
                               raw_data_path="datasets/RTDS/rtds_noisy_len100.hdf5")
        data_val = RTDS_data(split='valid',
                             mask_path=data_path,
                             return_Y=args.return_y,
                             return_Edge=False,
                             delta_t=0.01, 
                             noise_ratio=args.np, 
                             raw_data_path="datasets/RTDS/rtds_noisy_len100.hdf5")
        data_test = RTDS_data(split='test',
                              mask_path=data_path,
                              return_Y=args.return_y,
                              return_Edge=False,
                              delta_t=0.01, 
                              noise_ratio=args.np, 
                              raw_data_path="datasets/RTDS/rtds_noisy_len100.hdf5")
    else:
        data_train = None
        data_val = None
        data_test = None
    train = DataLoader(dataset=data_train,
                       shuffle=True,
                       batch_size=args.batch_size,
                       pin_memory=True)
    valid = DataLoader(dataset=data_val,
                       batch_size=args.batch_size,
                       pin_memory=True)
    test = DataLoader(dataset=data_test,
                      shuffle=False,
                      batch_size=args.batch_size,
                      pin_memory=True)
    return train, valid, test


def get_metrics(X_pred, X, Mask_v, mean, std):
    mean = mean.unsqueeze(1)
    std = std.unsqueeze(1)
    mse_v = (X_pred - X)**2
    mse_v = (mse_v * Mask_v).sum((1, 2, 3)) / (Mask_v.sum((1, 2, 3)) + 1e-6)
    mae_v = torch.abs(X_pred - X)
    mae_v = (mae_v * Mask_v).sum((1, 2, 3)) / (Mask_v.sum((1, 2, 3)) + 1e-6)
    mape_v = torch.atan(torch.abs(X_pred - X) / (torch.abs(X)))
    mape_v = (mape_v * Mask_v).sum((1, 2, 3)) / (Mask_v.sum((1, 2, 3)) + 1e-6)
    mape_v = torch.tan(mape_v)
    return mse_v, mae_v, mape_v


def train(args):
    log_folder = os.path.join("./logs", get_ckp_path(args)[1].split('/')[2])
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_path = os.path.join(log_folder, (get_ckp_path(args)[0].split('/')[3].split('.')[0] + '.' + get_ckp_path(args)[0].split('/')[3].split('.')[1] + ".log"))
    logger = get_logger(log_path)
    model = get_model(args)
    print (args.checkname)
    if args.checkname:
        print("Loading previous model.")
        model.load_state_dict(torch.load(args.checkname))
    worse_epochs, best_score = 0, float('inf')
    train, valid, test = get_dataset(args)
    # build dataset.
    if args.adversarial: 
        DisNet = Discriminator(args.dimIn, args.dimRnn, args.numRNNs,
                            args.numRamWalks).cuda()
        dis_path = get_ckp_path(args)[2]
        if os.path.exists(dis_path):
            print("Loading discriminator model.")
            DisNet.load_state_dict(torch.load(dis_path))
        optimizer_D = optim.RMSprop(DisNet.parameters(), lr=args.lr)
    else: 
        DisNet = None
        optimizer_D = None
    optimizer_G = optim.Adam(model.parameters(), lr=args.lr)
    # start training.
    for epoch in range(1, args.max_epoch + 1):
        """------Training------"""
        model.train()
        pbar = tqdm(train)
        pbar.write('\x1b[1;35mTraining Epoch\t%03d:\x1b[0m' % epoch)
        for n, data_batch in enumerate(pbar):
            data_batch = {
                key: value.cuda()
                for key, value in data_batch.items()
            }
            value, Mask = copy.deepcopy(data_batch['values']), copy.deepcopy(
                data_batch['masks'])
            """------Generator------"""
            optimizer_G.zero_grad()
            loss, X_pred = model.get_loss(data_batch)
            data_batch['values'], data_batch['masks'] = value, Mask
            if args.adversarial: 
                A_x = data_batch['adjacent_matrices']
                labels = torch.ones(size=(X_pred.size(0),
                                      X_pred.size(2))).cuda()
                loss_G = DisNet.forward(X_pred, Mask[:, 1:, :, :], A_x, labels)
                pbar.write(
                    'Epoch\t{:04d}, Iteration\t{:04d}: Generator loss\t{:6.4f};'
                    .format(epoch, n, loss_G))
                loss = loss + args.alpha * loss_G
            loss.backward()
            optimizer_G.step()
            pbar.write(
                'Epoch\t{:04d}, Iteration\t{:04d}: loss\t{:6.4f};'.format(
                    epoch, n, loss))
            if args.adversarial: 
                for _ in range(1):
                    optimizer_D.zero_grad()
                    # fake.
                    _, X_pred = model.get_loss(data_batch)
                    data_batch['values'], data_batch['masks'] = value, Mask
                    Mask = data_batch['masks']
                    A_x = data_batch['adjacent_matrices']
                    labels = torch.zeros(size=(X_pred.size(0),
                                               X_pred.size(2))).cuda()
                    loss_fake = -DisNet.forward(X_pred, Mask[:, 1:, :, :], A_x,
                                                labels)
                    loss_fake.backward()
                    # real.
                    X = data_batch['values'][:, 1:, :, :]
                    labels = torch.ones(size=(X_pred.size(0),
                                              X_pred.size(2))).cuda()
                    loss_real = DisNet.forward(X, Mask[:, 1:, :, :], A_x,
                                               labels)
                    loss_real.backward()
                    optimizer_D.step()
                    Loss_D = loss_real + loss_fake
                    pbar.write(
                        'Epoch\t{:04d}, Iteration\t{:04d}: Wasserstein Distance\t{:6.4f};'
                        .format(epoch, n, Loss_D * 10000))
        """------Validation------"""
        model.eval()
        pbar = tqdm(valid)
        pbar.write('\x1b[1;35mValidation Epoch\t%03d:\x1b[0m' % epoch)
        mse_v, mae_v, mape_v, numSeq = 0., 0., 0., 0.
        for n, data_batch in enumerate(pbar):
            data_batch = {
                key: value.cuda()
                for key, value in data_batch.items()
            }
            with torch.no_grad():
                X_pred = model.forward(data_batch)[0]
                X = data_batch['values'][:, 1:, :, :]
                mean, std = data_batch['mean'], data_batch['std']
                # generate mask to indicate the valid node positions.
                Mask_v = torch.zeros_like(X)
                for i, num_v in enumerate(data_batch['#nodes']):
                    Mask_v[i, :, :num_v, :] = 1.0
                if args.np != 0: 
                    mse_v_n, mae_v_n, mape_v_n = get_metrics(X_pred, data_batch['ori_values'][:, 1:, :, :],
                                                     Mask_v, mean, std) 
                else: 
                    mse_v_n, mae_v_n, mape_v_n = get_metrics(X_pred, X,
                                                         Mask_v, mean, std) 
                mse_v += mse_v_n.sum()
                mae_v += mae_v_n.sum()
                mape_v += mape_v_n.sum()
                numSeq += X.size(0)
        rmse_v = torch.sqrt(mse_v / numSeq).detach().cpu().numpy()
        mae_v = (mae_v / numSeq).detach().cpu().numpy()
        mape_v = (mape_v / numSeq).detach().cpu().numpy()
        pbar.write(
            '\x1b[1;35mValidation Epoch\t%03d: RMSE(Node)--> %6.4f; MAE(Node)-->%6.4f; MAPE(Node)-->%6.4f.\x1b[0m.\x1b[0m'
            % (epoch, rmse_v, mae_v, mape_v))
        logger.info('Epoch\t%03d: RMSE(Node)--> %6.4f; MAE(Node)-->%6.4f; MAPE(Node)-->%6.4f' % (epoch, rmse_v, mae_v, mape_v))
        score = (rmse_v + mae_v) / 2.
        # early stopping and save models.
        if score < best_score:
            best_score = score
            worse_epochs = 0
            if args.saveto:
                saveto_path = get_ckp_path(args)[0]
                torch.save(model.state_dict(), saveto_path)
        else:
            worse_epochs += 1
        #
        if worse_epochs >= args.early_stop:#
            break
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    return


def test(args):
    log_folder = os.path.join("./logs", get_ckp_path(args)[1].split('/')[2])
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_path = os.path.join(log_folder, (get_ckp_path(args)[0].split('/')[3].split('.')[0] + '.' + get_ckp_path(args)[0].split('/')[3].split('.')[1] + ".log"))
    logger = get_logger(log_path)
    model = get_model(args)
    if args.checkname:
        print("Loading previous model." + args.checkname)
        model.load_state_dict(torch.load(args.checkname))
    # build dataset.
    _, _, test = get_dataset(args)
    pbar = tqdm(test)
    pbar.write('\x1b[1;35mTesting:\x1b[0m')
    mse_v, mae_v, mape_v, numSeq = 0., 0., 0., 0.
    for n, data_batch in enumerate(pbar):
        data_batch = {key: value.cuda() for key, value in data_batch.items()}
        with torch.no_grad():
            X_pred = model.forward(data_batch)[0]
            X = data_batch['values'][:, 1:, :, :]
            mean, std = data_batch['mean'], data_batch['std']
            Mask_v = torch.zeros_like(X)
            for i, num_v in enumerate(data_batch['#nodes']):
                Mask_v[i, :, :num_v, :] = 1.0
            if args.np != 0: 
                mse_v_n, mae_v_n, mape_v_n = get_metrics(X_pred, data_batch['ori_values'][:, 1:, :, :],
                                                     Mask_v, mean, std)  
            else: 
                mse_v_n, mae_v_n, mape_v_n = get_metrics(X_pred, X,
                                                     Mask_v, mean, std)
            mse_v += mse_v_n.sum()
            mae_v += mae_v_n.sum()
            mape_v += mape_v_n.sum()
            numSeq += X.size(0)
    rmse_v = torch.sqrt(mse_v / numSeq).detach().cpu().numpy()
    mae_v = (mae_v / numSeq).detach().cpu().numpy()
    mape_v = (mape_v / numSeq).detach().cpu().numpy()
    pbar.write(
        '\x1b[1;35mTesting: RMSE(Node)--> %6.4f; MAE(Node)-->%6.4f; MAPE(Node)-->%6.4f.\x1b[0m'
        % (rmse_v, mae_v, mape_v))
    logger.info('Testing ps - %6.4f, pt - %6.4f: RMSE(Node)--> %6.4f; MAE(Node)-->%6.4f; MAPE(Node)-->%6.4f' % (args.p_spatial, args.p_temporal, rmse_v, mae_v, mape_v))
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    return
