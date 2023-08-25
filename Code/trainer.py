"""
Author: Yingru Liu.
"""

import time
import logging 
import argparse, os, copy
import torch
import torch.optim as optim
import scipy.io as sio
import numpy as np
from tqdm.auto import tqdm
from processing import RTDS_data, Traffic_data, Motion_data, RTDSReal_data, BCM_data
from torch.utils.data import DataLoader
from models import (STGCN, GCGRUODE, DCGRUODE, DDCGRUODE, GCODERNN, DCODERNN, DDCODERNN, GCGRU,
                    DCGRU, DDCGRU, GRU, NGDCSDE, HGDCODE, NODE, HNODE, Discriminator, FFNN, LSTM)#, GraphVSDN, VSDN_ODE, GNODE, GRAPHODE, GGRAPHODE)

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
    #sub_folder = '{}_spatial{}_temporal{}_downsampling{}'.format(
    #    args.dataset, int(args.p_spatial // 0.0999),
    #    int(args.p_temporal // 0.0999), args.downsampling)
    #sub_folder = '{}_missing{}_downsampling{}'.format(
    #    args.dataset, int(args.p // 0.0999), args.downsampling)
    if (args.dataset.lower() == 'rtds_real' or args.dataset.lower() == 'rtds_noisy' or args.dataset.lower() == 'rtds_nodes' or args.dataset.lower() in ['bcm', 'bcm_ss2_load', 'bcm_ss2_bus', 'bcm_ss1_load', 'bcm_ss1_bus', 'bcm_load', 'bcm_connected', 'bcm_islanded1', 'bcm_islanded2']):
        sub_folder = '{}_len{}_p{}_missing{}_noise{}_downsampling{}'.format(
        args.dataset, args.nf, args.dr, int(args.p // 0.0999), int(args.np // 0.00999), args.downsampling)
    else: 
        sub_folder = '{}_spatial{}_temporal{}_downsampling{}'.format(
        args.dataset, int(args.p_spatial // 0.0999),
        int(args.p_temporal // 0.0999), args.downsampling)
    if args.model_name.lower() == 'stgcn':
        file_name = '{}_conva{}_conb{}_kernela{}_#block{}.pth'.format(
            args.model_name, args.dimOutConv1, args.dimOutConv2,
            args.kernel_size, args.numBlock)
    elif args.model_name.lower() == 'ffnn':
        file_name = '{}_vHidden{}_#numHidden{}.pth'.format(
            args.model_name, args.dimODEHidden, 2)
    elif args.model_name.lower() == 'lstm':
        file_name = '{}_vRnn{}_#numRNNs{}.pth'.format(
            args.model_name, args.dimRnn, args.numRNNs)
    elif args.model_name.lower() == 'gru':
        file_name = '{}_vRnn{}_#numRNNs{}.pth'.format(
            args.model_name, args.dimRnn, args.numRNNs)
    elif args.model_name.lower() == 'gcgru':
        file_name = '{}_vRnn{}_#numRNNs{}.pth'.format(
            args.model_name, args.dimRnn, args.numRNNs)
    elif args.model_name.lower() == 'dcgru':
        file_name = '{}_vRnn{}_#numRNNs{}_#randomWalk.pth'.format(
            args.model_name, args.dimRnn, args.numRNNs, args.numRamWalks)
    elif args.model_name.lower() == 'ddcgru':
        file_name = '{}_vRnn{}_#numRNNs{}_#randomWalk.pth'.format(
            args.model_name, args.dimRnn, args.numRNNs, args.numRamWalks)
    elif args.model_name.lower() == 'node':
        file_name = '{}_vRnn{}_ODEHidden{}_#ODEHidden{}_deltaT{}_#randomWalk{}_{}.pth'.format(
            args.model_name, args.dimRnn, args.dimODEHidden,
            args.numODEHidden, args.delta_t, args.numRamWalks, args.odeMethod)
    elif args.model_name.lower() == 'gcodernn':
        file_name = '{}_vRnn{}_#numRNNs{}_ODEHidden{}_#ODEHidden{}_deltaT{}_{}.pth'.format(
            args.model_name, args.dimRnn, args.numRNNs, args.dimODEHidden,
            args.numODEHidden, args.delta_t, args.odeMethod)
    elif args.model_name.lower() == 'dcodernn':
        file_name = '{}_vRnn{}_#numRNNs{}_ODEHidden{}_#ODEHidden{}_deltaT{}_#randomWalk{}_{}.pth'.format(
            args.model_name, args.dimRnn, args.numRNNs, args.dimODEHidden,
            args.numODEHidden, args.delta_t, args.numRamWalks, args.odeMethod)
    elif args.model_name.lower() == 'ddcodernn':
        file_name = '{}_vRnn{}_#numRNNs{}_ODEHidden{}_#ODEHidden{}_deltaT{}_#randomWalk{}_{}.pth'.format(
            args.model_name, args.dimRnn, args.numRNNs, args.dimODEHidden,
            args.numODEHidden, args.delta_t, args.numRamWalks, args.odeMethod)
    elif args.model_name.lower() == 'gcgruode':
        file_name = '{}_vRnn{}_#numRNNs{}_deltaT{}_{}.pth'.format(
            args.model_name, args.dimRnn, args.numRNNs, args.delta_t, args.odeMethod)
    elif args.model_name.lower() == 'dcgruode':
        file_name = '{}_vRnn{}_#numRNNs{}_deltaT{}_#randomWalk{}_{}.pth'.format(
            args.model_name, args.dimRnn, args.numRNNs, args.delta_t,
            args.numRamWalks, args.odeMethod)
    elif args.model_name.lower() == 'ddcgruode':
        file_name = '{}_vRnn{}_#numRNNs{}_deltaT{}_#randomWalk{}_{}.pth'.format(
            args.model_name, args.dimRnn, args.numRNNs, args.delta_t,
            args.numRamWalks, args.odeMethod)
    elif args.model_name.lower() == 'hgdcode' or args.model_name.lower() == 'hnode':
        file_name = '{}_vRnn{}_vLatent{}_ODEHidden{}_#ODEHidden{}_deltaT{}_#randomWalk{}_{}.pth'.format(
            args.model_name, args.dimRnn, args.dimState, args.dimODEHidden,
            args.numODEHidden, args.delta_t, args.numRamWalks, args.odeMethod)
    elif args.model_name.lower() == 'ngdcsde':
        file_name = '{}_{}_vRnn{}_vLatent{}_#numRNNs{}_deltaT{}_lambda{}_#randomWalk{}_#traj{}_{}.pth'.format(
            args.model_name, args.sdeConnect, args.dimRnn, args.dimState, args.numRNNs,
            args.delta_t, int(args.beta // 0.00999), args.numRamWalks, args.numTraj, args.adversarial)
    elif args.model_name.lower() == 'graphvsdn':
        file_name = 'graphvsdn-testing.pth'
    elif args.model_name.lower() == 'vsdnode':
        file_name = 'vsdnode-testing.pth'
    elif args.model_name.lower() == 'gru': 
        file_name = '{}_vRnn{}_#numRNNs{}_deltaT{}.pth'.format(
            args.model_name, args.dimRnn, args.numRNNs, args.delta_t)
    else:
        file_name = 'unknown_model.pth'
    discriminator_name = '{}_discriminator.pth'.format(args.model_name)
    sub_path = os.path.join(main_path, sub_folder)
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
    #
    return os.path.join(sub_path, file_name), sub_path, os.path.join(sub_path, discriminator_name)


def get_model(args):
    if args.model_name.lower() == 'stgcn':
        return STGCN(args.dimIn,
                     args.dimOutConv1,
                     args.dimOutConv2,
                     args.kernel_size,
                     args.dimGCN,
                     args.numBlock,
                     learnstd=args.learnstd).cuda()
    elif args.model_name.lower() == 'ffnn':
        return FFNN(args.dimIn,
                   args.dimODEHidden,
                   2,
                   args.numNode,
                   learnstd=args.learnstd).cuda()
    elif args.model_name.lower() == 'lstm':
        return LSTM(args.dimIn,
                   args.dimRnn,
                   args.numRNNs,
                   args.numNode,
                   learnstd=args.learnstd).cuda()
    elif args.model_name.lower() == 'gru':
        return GRU(args.dimIn,
                   args.dimRnn,
                   args.numRNNs,
                   args.numNode,
                   learnstd=args.learnstd).cuda()
    elif args.model_name.lower() == 'gcgru':
        return GCGRU(args.dimIn,
                     args.dimRnn,
                     args.numRNNs,
                     learnstd=args.learnstd).cuda()
    elif args.model_name.lower() == 'dcgru':
        return DCGRU(args.dimIn,
                     args.dimRnn,
                     args.numRNNs,
                     args.numRamWalks,
                     learnstd=args.learnstd).cuda()
    elif args.model_name.lower() == 'ddcgru':
        return DDCGRU(args.dimIn,
                      args.dimRnn,
                      args.numRNNs,
                      args.numNode, 
                      args.numRamWalks,
                      learnstd=args.learnstd).cuda()
    elif args.model_name.lower() == 'node':
        return NODE(args.dimIn,
                    args.dimRnn,
                    args.dimState,
                    args.dimODEHidden,
                    args.numODEHidden,
                    args.delta_t,
                    args.beta,
                    args.numRamWalks,
                    learnstd=args.learnstd,
                    ode_method = args.odeMethod).cuda()
    elif args.model_name.lower() == 'gcodernn':
        return GCODERNN(args.dimIn,
                        args.dimRnn,
                        args.dimState,
                        args.dimODEHidden,
                        args.numODEHidden,
                        args.delta_t,
                        args.beta,
                        args.numRamWalks,
                        learnstd=args.learnstd,
                        ode_method = args.odeMethod).cuda()
    elif args.model_name.lower() == 'dcodernn':
        return DCODERNN(args.dimIn,
                        args.dimRnn,
                        args.dimState,
                        args.dimODEHidden,
                        args.numODEHidden,
                        args.delta_t,
                        args.beta,
                        args.numRamWalks,
                        learnstd=args.learnstd,
                        ode_method = args.odeMethod).cuda()
    elif args.model_name.lower() == 'ddcodernn':
        return DDCODERNN(args.dimIn,
                         args.dimRnn,
                         args.dimState,
                         args.dimODEHidden,
                         args.numODEHidden,
                         args.numNode, 
                         args.delta_t,
                         args.beta,
                         args.numRamWalks,
                         learnstd=args.learnstd,
                         ode_method = args.odeMethod).cuda()
    elif args.model_name.lower() == 'gcgruode':
        return GCGRUODE(args.dimIn,
                        args.dimRnn,
                        args.dimState,
                        args.dimODEHidden,
                        args.numODEHidden,
                        args.delta_t,
                        args.beta,
                        args.numRamWalks,
                        learnstd=args.learnstd,
                        ode_method = args.odeMethod).cuda()
    elif args.model_name.lower() == 'dcgruode':
        return DCGRUODE(args.dimIn,
                        args.dimRnn,
                        args.dimState,
                        args.dimODEHidden,
                        args.numODEHidden,
                        args.delta_t,
                        args.beta,
                        args.numRamWalks,
                        learnstd=args.learnstd,
                        ode_method = args.odeMethod).cuda()
    elif args.model_name.lower() == 'ddcgruode':
        return DDCGRUODE(args.dimIn,
                         args.dimRnn,
                         args.dimState,
                         args.dimODEHidden,
                         args.numODEHidden,
                         args.numNode, 
                         args.delta_t,
                         args.beta,
                         args.numRamWalks,
                         learnstd=args.learnstd,
                         ode_method = args.odeMethod).cuda()
    elif args.model_name.lower() == 'hgdcode':
        return HGDCODE(args.dimIn,
                       args.dimRnn,
                       args.dimState,
                       args.dimODEHidden,
                       args.numODEHidden, 
                       args.numNode, 
                       args.delta_t,
                       args.beta,
                       args.numRamWalks,
                       args.learnstd,
                       args.odeMethod).cuda()
    elif args.model_name.lower() == 'ngdcsde':
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
                       args.learnstd, 
                       args.odeMethod, 
                       args.sdeMethod,
                       args.sdeConnect).cuda()
    elif args.model_name.lower() == 'graphvsdn':
        return GraphVSDN(args.dimIn,
                         args.dimODEHidden,
                         args.numODEHidden,
                         args.dimODEHidden,
                         dim_rnn=args.dimRnn,
                         dim_state=args.dimState,
                         delta_t=args.delta_t,
                         Lambda=args.beta,
                         numSample=args.numTraj).cuda()#,
                         #encoding='gc').cuda()
    elif args.model_name.lower() == 'vsdnode':
        return VSDN_ODE(args.dimIn,
                         args.dimODEHidden,
                         args.numODEHidden,
                         args.dimODEHidden,
                         dim_rnn=args.dimRnn,
                         dim_state=args.dimState,
                         delta_t=args.delta_t,
                         Lambda=args.beta,
                         numSample=args.numTraj).cuda()
    return


def get_dataset(args):
    if args.dataset.lower() == 'rtds_real': 
        filename = '{}_len{}_p{}_missing{}_downsampling{}.hdf5'.format(
            'rtds', args.nf, args.dr, int(args.p // 0.0999), args.downsampling)
        data_path = os.path.join(args.data_path, filename)
        data_train = RTDSReal_data(split='train',
                               mask_path=data_path,
                               return_Y=args.return_y,
                               delta_t=args.delta_t, 
                               noise_ratio=args.np, 
                              data_rate = args.dr)
        data_val = RTDSReal_data(split='valid',
                             mask_path=data_path,
                             return_Y=args.return_y,
                             delta_t=args.delta_t, 
                             noise_ratio=args.np, 
                              data_rate = args.dr)
        data_test = RTDSReal_data(split='test',
                              mask_path=data_path,
                              return_Y=args.return_y,
                              delta_t=args.delta_t, 
                              noise_ratio=args.np, 
                              data_rate = args.dr)
    elif args.dataset.lower() in ['bcm', 'bcm_ss2_load', 'bcm_ss2_bus', 'bcm_ss1_load', 'bcm_ss1_bus', 'bcm_load', 'bcm_connected', 'bcm_islanded1', 'bcm_islanded2']: 
        filename = '{}_len{}_p{}_missing{}_downsampling{}.hdf5'.format(
            'bcm', args.nf, args.dr, int(args.p // 0.0999), args.downsampling)
        data_path = os.path.join(args.data_path, filename)
        data_train = BCM_data(split='train',
                               mask_path=data_path,
                               return_Y=args.return_y,
                               delta_t=args.delta_t, 
                               noise_ratio=args.np, 
                              data_rate = args.dr)
        data_val = BCM_data(split='valid',
                             mask_path=data_path,
                             return_Y=args.return_y,
                             delta_t=args.delta_t, 
                             noise_ratio=args.np, 
                              data_rate = args.dr)
        data_test = BCM_data(split='test',
                              mask_path=data_path,
                              return_Y=args.return_y,
                              delta_t=args.delta_t, 
                              noise_ratio=args.np, 
                              data_rate = args.dr)
    elif args.dataset.lower() == 'rtds_noisy': 
        filename = '{}_len{}_missing{}_downsampling{}.hdf5'.format(
            'rtds_noisy', args.nf, int(args.p // 0.0999), args.downsampling)
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
    elif args.dataset.lower() == 'rtds_noisy_node': 
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
    elif args.dataset.lower() == 'rtds_noisy_edge': 
        filename = '{}_len{}_missing_points_spatial{}_temporal{}_downsampling{}.hdf5'.format(
            'rtds_noisy', args.nf, int(args.p_spatial // 0.0999),
            int(args.p_temporal // 0.0999), args.downsampling)
        data_path = os.path.join(args.data_path, filename)
        data_train = RTDS_data(split='train',
                               mask_path=data_path,
                               return_Y=args.return_y,
                               return_Edge=True,
                               delta_t=0.01, 
                               noise_ratio=args.np, 
                               raw_data_path="datasets/RTDS/rtds_noisy_len100.hdf5")
        data_val = RTDS_data(split='valid',
                             mask_path=data_path,
                             return_Y=args.return_y,
                             return_Edge=True,
                             delta_t=0.01, 
                             noise_ratio=args.np, 
                             raw_data_path="datasets/RTDS/rtds_noisy_len100.hdf5")
        data_test = RTDS_data(split='test',
                              mask_path=data_path,
                              return_Y=args.return_y,
                              return_Edge=True,
                              delta_t=0.01, 
                              noise_ratio=args.np, 
                              raw_data_path="datasets/RTDS/rtds_noisy_len100.hdf5")
    elif args.dataset.lower() == 'rtds_nodes':
        #filename = '{}_missing_points_spatial{}_temporal{}_downsampling{}.hdf5'.format(
        #    'rtds_data_100', int(args.p_spatial // 0.0999),
        #    int(args.p_temporal // 0.0999), args.downsampling)
        filename = '{}_len{}_missing{}_downsampling{}.hdf5'.format(
            'rtds', args.nf, int(args.p // 0.0999), args.downsampling)
        data_path = os.path.join(args.data_path, filename)
        data_train = RTDS_data(split='train',
                               mask_path=data_path,
                               return_Y=args.return_y,
                               return_Edge=False,
                               delta_t=args.delta_t, 
                               noise_ratio=args.np)
        data_val = RTDS_data(split='valid',
                             mask_path=data_path,
                             return_Y=args.return_y,
                             return_Edge=False,
                             delta_t=args.delta_t, 
                             noise_ratio=args.np)
        data_test = RTDS_data(split='test',
                              mask_path=data_path,
                              return_Y=args.return_y,
                              return_Edge=False,
                              delta_t=args.delta_t, 
                              noise_ratio=args.np)
    elif args.dataset.lower() == 'rtds_edges':
        filename = '{}_len{}_missing{}_downsampling{}.hdf5'.format(
            'rtds', args.nf, int(args.p // 0.0999), args.downsampling)
        data_path = os.path.join(args.data_path, filename)
        data_train = RTDS_data(split='train',
                               mask_path=data_path,
                               return_Y=args.return_y,
                               return_Edge=True,
                               delta_t=args.delta_t)
        data_val = RTDS_data(split='valid',
                             mask_path=data_path,
                             return_Y=args.return_y,
                             return_Edge=True,
                             delta_t=args.delta_t)
        data_test = RTDS_data(split='test',
                              mask_path=data_path,
                              return_Y=args.return_y,
                              return_Edge=True,
                              delta_t=args.delta_t)
    elif args.dataset.lower() in ['pems', 'metr']:
        filename = '{}_spatial{}_temporal{}.hdf5'.format(
            args.dataset.lower(), int(args.p_spatial // 0.0999),
            int(args.p_temporal // 0.0999))
        data_path = os.path.join(args.data_path, filename)
        data_train = Traffic_data(split='train',
                                  data_path=data_path,
                                  delta_t=args.delta_t)
        data_val = Traffic_data(split='valid',
                                data_path=data_path,
                                delta_t=args.delta_t)
        data_test = Traffic_data(split='test',
                                 data_path=data_path,
                                 delta_t=args.delta_t)
    elif args.dataset.lower() == 'h36m':
        filename = '{}_spatial{}_temporal{}.hdf5'.format(
            args.dataset.lower(), int(args.p_spatial // 0.0999),
            int(args.p_temporal // 0.0999))
        data_path = os.path.join(args.data_path, filename)
        data_train = Motion_data(split='train',
                                 data_path=data_path,
                                 delta_t=args.delta_t,
                                 p_spatial=args.p_spatial,
                                 p_temporal=args.p_temporal)
        data_val = Motion_data(split='valid',
                               data_path=data_path,
                               delta_t=args.delta_t,
                               p_spatial=args.p_spatial,
                               p_temporal=args.p_temporal)
        data_test = Motion_data(split='test',
                                data_path=data_path,
                                delta_t=args.delta_t,
                                p_spatial=args.p_spatial,
                                p_temporal=args.p_temporal)
    else:
        data_train = None
        data_val = None
        data_test = None
    train = DataLoader(dataset=data_train,
                       shuffle=True,
                       batch_size=args.batch_size,
                       pin_memory=False)#True)
    valid = DataLoader(dataset=data_val,
                       batch_size=args.batch_size,
                       pin_memory=False)#True)
    test = DataLoader(dataset=data_test,
                      shuffle=False,
                      batch_size=args.batch_size,
                      pin_memory=False)#True)
    return train, valid, test


def get_metrics(X_pred, X, Mask_v, mean, std):
    #print(X_pred.shape, X.shape)
    #X_pred = X_pred[:, 1:, :, :]
    #X = X[:, 1:, :, :]
    #Mask_v = Mask_v[:, 1:, :, :]
    mean = mean.unsqueeze(1)
    std = std.unsqueeze(1)
    #print(mean)
    #print(mean.shape, std.shape, X.shape)
    mse_v = (X_pred - X)**2
    mse_v = (mse_v * Mask_v).sum((1, 2, 3)) / (Mask_v.sum((1, 2, 3)) + 1e-6)
    mae_v = torch.abs(X_pred - X)
    mae_v = (mae_v * Mask_v).sum((1, 2, 3)) / (Mask_v.sum((1, 2, 3)) + 1e-6)
    
    #mape_v = torch.abs(X_pred - X) / (torch.abs(X) + 1e-6)
    #####mape_v = torch.abs(X_pred * std - X * std) / (torch.abs(X * std + mean) + 1e-6)#1e-12
    #mape_v = (torch.abs(X_pred - X) / torch.abs(X)) * (torch.abs(X) > 1e-12) + 1.0 * ((torch.abs(X) < 1e-12) * (torch.abs(X_pred - X) != 0)) + 0.0 * ((torch.abs(X) < 1e-12) * (torch.abs(X_pred - X) == 0))
    #mape_v = torch.abs(X_pred - X) / ((torch.abs(X) * (torch.abs(X) != 0)) + (torch.abs(X_pred - X) * (torch.abs(X) == 0)))
    #mape_v = (mape_v[:, :, 1:, :] * Mask_v[:, :, 1:, :]).sum((1, 2, 3)) / (Mask_v[:, :, 1:, :].sum((1, 2, 3)) + 1e-6)
    #mape_v = (mape_v * Mask_v).sum((1, 2, 3)) / (Mask_v.sum((1, 2, 3)) + 1e-6)

    '''
    # WMAPE
    mape_v1 = (torch.abs(X_pred - X) * Mask_v).sum((1, 2, 3))
    mape_v2 = (torch.abs(X) * Mask_v).sum((1, 2, 3))
    mape_v = (mape_v1 / mape_v2)# / (Mask_v.sum((1, 2, 3)) + 1e-6)
    '''
    '''
    # SMAPE
    mape_v = torch.abs(X_pred - X) / ((torch.abs(X) + torch.abs(X_pred)) / 2 + 1e-12) 
    mape_v = (mape_v * Mask_v).sum((1, 2, 3)) / (Mask_v.sum((1, 2, 3)) + 1e-6)
    '''
    
    
    # MAAPE
    mape_v = torch.atan(torch.abs(X_pred - X) / (torch.abs(X)))
    #mape_v = torch.atan(torch.abs(X_pred * std - X * std) / torch.abs(X * std + mean))
    mape_v = (mape_v * Mask_v).sum((1, 2, 3)) / (Mask_v.sum((1, 2, 3)) + 1e-6)
    mape_v = torch.tan(mape_v)
    return mse_v, mae_v, mape_v


def get_metrics1(X_pred, X, Mask_v, mean, std):
    #print(X_pred.shape, X.shape)
    #X_pred = X_pred[:, 1:, :, :]
    #X = X[:, 1:, :, :]
    #Mask_v = Mask_v[:, 1:, :, :]
    mean = mean.unsqueeze(1)
    std = std.unsqueeze(1)
    #print(mean)
    #print(mean.shape, std.shape, X.shape)
    mse_v = (X_pred - X)**2
    mse_v = (mse_v * Mask_v).sum((1, 2, 3)) / (Mask_v.sum((1, 2, 3)) + 1e-6)
    mae_v = torch.abs(X_pred - X)
    mae_v = (mae_v * Mask_v).sum((1, 2, 3)) / (Mask_v.sum((1, 2, 3)) + 1e-6)
    
    mape_v = torch.abs(X_pred - X) / (torch.abs(X) + 1e-6)
    #####mape_v = torch.abs(X_pred * std - X * std) / (torch.abs(X * std + mean) + 1e-6)#1e-12
    #mape_v = (torch.abs(X_pred - X) / torch.abs(X)) * (torch.abs(X) > 1e-12) + 1.0 * ((torch.abs(X) < 1e-12) * (torch.abs(X_pred - X) != 0)) + 0.0 * ((torch.abs(X) < 1e-12) * (torch.abs(X_pred - X) == 0))
    #mape_v = torch.abs(X_pred - X) / ((torch.abs(X) * (torch.abs(X) != 0)) + (torch.abs(X_pred - X) * (torch.abs(X) == 0)))
    #mape_v = (mape_v[:, :, 1:, :] * Mask_v[:, :, 1:, :]).sum((1, 2, 3)) / (Mask_v[:, :, 1:, :].sum((1, 2, 3)) + 1e-6)
    mape_v = (mape_v * Mask_v).sum((1, 2, 3)) / (Mask_v.sum((1, 2, 3)) + 1e-6)

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
        model.load_state_dict(torch.load(args.checkname))#, strict = False)
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
    '''
    if args.model_name.lower() == 'ngdcsde':# or args.model_name.lower() == 'ngdcsde1': 
        optimizer_G = optim.Adam([{"params": model.Encoder_pre.parameters()}, 
                                  {"params": model.Encoder_post.parameters()}, 
                                  {"params": model.Net_diff.parameters()}, 
                                  {"params": model.Net_drift.parameters()}, 
                                  {"params": model.Net_prior.parameters()}, 
                                  {"params": model.Net_post.parameters()}, 
                                  {"params": model.Output.parameters()},  
                                  {"params": model.z0}, 
                                  {"params": model.ODE.parameters(), "lr": args.lr}], lr=args.lr * 0.5)
    else: 
    '''
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
                                                     Mask_v, mean, std)  #,mape_v_n data_batch['masks'][:, 1:, :, :])#
                else: 
                    mse_v_n, mae_v_n, mape_v_n = get_metrics(X_pred, X,
                                                         Mask_v, mean, std)  # ,mape_v_n
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
    #log_path = os.path.join(args.logname, (args.model_name + ".log"))
    logger = get_logger(log_path)
    model = get_model(args)
    if args.checkname:
        print("Loading previous model." + args.checkname)
        model.load_state_dict(torch.load(args.checkname))#, strict = False)
    model.eval()
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
            #print(mean[0].shape, std[0].shape)
            #np.save('mean2.npy', mean[0].cpu())
            #np.save('std2.npy', std[0].cpu())
            #return
            # generate mask to indicate the valid node positions.
            Mask_v = torch.zeros_like(X)
            for i, num_v in enumerate(data_batch['#nodes']):
                Mask_v[i, :, :num_v, :] = 1.0
            if args.np != 0: 
                mse_v_n, mae_v_n, mape_v_n = get_metrics(X_pred, data_batch['ori_values'][:, 1:, :, :],
                                                     Mask_v, mean, std)  #,mape_v_n data_batch['masks'][:, 1:, :, :])#
                #print(mape_v_n)
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


def predict(args):
    model = get_model(args)
    if args.checkname:
        print("Loading previous model.")
        print(args.checkname)
    model.load_state_dict(torch.load(args.checkname))
    model.eval()
    _, _, test = get_dataset(args)
    pbar = tqdm(test)
    pbar.write('\x1b[1;35mTesting:\x1b[0m')
    XS, XMS, XPS, TPS, Mean = [], [], [], [], []
    for n, data_batch in enumerate(pbar):
        data_batch = {key: value.cuda() for key, value in data_batch.items()}
        mean, std = data_batch['mean'], data_batch['std']
        with torch.no_grad():
            X = data_batch['values']
            Xm = X * data_batch['masks']
            A = data_batch['adjacent_matrices']
            Delta_T = data_batch['t'][0][1] - data_batch['t'][0][0]
            tk = 1
            ti, tn = 0, data_batch['t'].size(1)
            Xp, tp = [], []
            h, g, z = None, None, None
            while ti < tn - 1:
                if ti % tk == 0:
                    start = time.time()
                    pred, _, h, g, z = model.predict(Xm[:, ti, :, :], A, args.delta_t, Delta_T, h, g, z)
                    end = time.time()
                    #print('Running time: %s ms' % ((end - start) * 1000))
                else:
                    pred, _, h, g, z = model.predict(None, A, args.delta_t, Delta_T, h, g, z)
                Xp.append((pred * std + mean).squeeze(0).cpu().numpy())
                tp.append(data_batch['t'][0][ti].squeeze(0).cpu().numpy())
                ti += 1
            Xp = np.array(Xp)
            tp = np.array(tp)
            X = X * std + mean
            Xm = Xm * std + mean
            XS.append(X[:, 1:, :, :].squeeze(0).cpu().numpy())
            XMS.append(Xm[:, 1:, :, :].squeeze(0).cpu().numpy())
            XPS.append(Xp)
            TPS.append(tp)
            Mean.append(mean.squeeze(0).cpu().numpy())
        if n >= 1:
            break
    XS = np.array(XS)
    XMS = np.array(XMS)
    XPS = np.array(XPS)
    TPS = np.array(TPS)
    Mean = np.array(Mean)
    sio.savemat('result1.mat', {'X' : XS, 'Xm' : XMS, 'Xp' : XPS, 'Tp' : TPS, 'Mean' : Mean})
    return


def plot(args): 
    args.model_name = 'DDCGRU'#'HGDCODE'
    args.checkname = get_ckp_path(args)[0]
    model1 = get_model(args)
    if args.checkname: 
        print("Loading previous model.")
        print(args.checkname)
        model1.load_state_dict(torch.load(args.checkname))#, strict = False)
    model1.eval()
    # build dataset.
    #args.p_spatial = 0.9
    #args.p_temporal = 0.7
    args.adversarial = True
    args.model_name = 'HGDCODE'
    args.checkname = get_ckp_path(args)[0]
    model2 = get_model(args)
    if args.checkname: 
        print("Loading previous model.")
        print(args.checkname)
        model2.load_state_dict(torch.load(args.checkname))
    _, _, test = get_dataset(args)
    pbar = tqdm(test)
    pbar.write('\x1b[1;35mTesting:\x1b[0m')
    Xps1, Xps2, Xs, Xms, Adj, Mean = [], [], [], [], [], []
    for n, data_batch in enumerate(pbar):
        data_batch = {key: value.cuda() for key, value in data_batch.items()}
        #print(data_batch['mean'].shape)
        mean, std = data_batch['mean'], data_batch['std']
        with torch.no_grad():
            #aa = copy.deepcopy(data_batch['adjacent_matrices'])
            #data_batch['adjacent_matrices'] = torch.zeros_like(aa)
            X_pred1 = model1.forward(data_batch)[0]# * std + mean
            X_pred2 = model2.forward(data_batch)[0]# * std + mean #data_batch['ori_values'][:, 1:, :, :]#
            X = data_batch['values'][:, 1:, :, :]# * std + mean
            Xm = (data_batch['values'] * data_batch['masks'])[:, 1:, :, :]# * std + mean
            #freq = data_batch['masks'].sum((1))[0, :, 0]
            #print(freq)
            #data_batch['adjacent_matrices'] = aa
            Xps1.append(X_pred1.squeeze(0).cpu().numpy())
            Xps2.append(X_pred2.squeeze(0).cpu().numpy())
            Xs.append(X.squeeze(0).cpu().numpy())
            Xms.append(Xm.squeeze(0).cpu().numpy())
            Adj.append(data_batch['adjacent_matrices'].squeeze(0).cpu().numpy())
            Mean.append(mean.squeeze(0).cpu().numpy())
            if n > 10: 
                break
    Xps1 = np.array(Xps1)
    Xps2 = np.array(Xps2)
    Xs = np.array(Xs)
    Xms = np.array(Xms)
    Adj = np.array(Adj)
    Mean = np.array(Mean)
    sio.savemat('result.mat', {'X' : Xs, 'Xm' : Xms, 'Xp1' : Xps1, 'Xp2' : Xps2, 'Adj' : Adj, 'Mean' : Mean})
    #sio.savemat('result.mat', {'X' : Xs, 'Xm' : Xms, 'Xp' : Xps1, 'Adj' : Adj, 'Mean' : Mean})
    return