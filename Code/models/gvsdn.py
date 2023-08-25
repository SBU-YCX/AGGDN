"""
Author: Yingru Liu
Graph-VSDN.
"""

import torch
import torch.nn as nn
from models.modules import MaskedMSEloss, GCGRUCell, DCGRUCell
from models.modules.mlp import GCNet, DCNet, DensNet
from models.vsdn.NeuralODE import LatentODE
from models.vsdn.VSDN_IWAE import VSDN_IWAE_FILTER, VSDN_IWAE_SMOOTH
from models.vsdn.VSDN_VAE import VSDN_VAE_FILTER, VSDN_VAE_SMOOTH


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, x, A):
        return x


class GraphVSDN(nn.Module):
    def __init__(self,
                 enc_dimIn,
                 enc_dimHidden,
                 enc_numHidden,
                 enc_dimOut,
                 dim_rnn,
                 dim_state,
                 delta_t,
                 Lambda,
                 numSample,
                 enc_activation=nn.ReLU,
                 encoding='identity'):
        super().__init__()
        assert encoding in ['identity', 'dc', 'gc'], "Incorrect Encoder type."
        if encoding == 'gc':
            self.encoder = GCNet(enc_dimIn,
                                 enc_dimHidden,
                                 enc_numHidden,
                                 enc_dimOut,
                                 activation=enc_activation)
        elif encoding == 'dc':
            self.encoder = DCNet(enc_dimIn,
                                 enc_dimHidden,
                                 enc_numHidden,
                                 enc_dimOut,
                                 activation=enc_activation)
        else:
            self.encoder = Identity()
        # TODO:
        self.vsdn = VSDN_VAE_FILTER(enc_dimIn,#enc_dimOut,
                                    dim_state,
                                    dim_rnn,
                                    enc_dimIn,
                                    enc_dimHidden,
                                    enc_dimHidden,
                                    delta_t,
                                    None,
                                    Lambda=Lambda,
                                    default_num_samples=numSample)
        self.delta_t = delta_t
        self.numSample = numSample
        return

    def forward(self, data_batch, delta_t=None, numSample=None, return_h=True):
        if delta_t is None:
            delta_t = self.delta_t
        if numSample is None:
            numSample = self.numSample
        t = data_batch['t'][0]
        Mask = data_batch['masks']
        X = data_batch['values'] * Mask
        A_xt = data_batch['adjacent_matrices']
        #
        encoding = self.encoder(X, A_xt).transpose(0, 1)
        time_len, batch_size, node_num, feature_size = encoding.shape
        X = X.transpose(0, 1)#.reshape(time_len, batch_size * node_num, -1)
        Mask = Mask.transpose(0, 1)#.reshape(time_len, batch_size * node_num, -1)
        cov = torch.zeros(size=(*X.shape[1:-1], 1)).cuda()
        _, loss, loss_recon, loss_kld, path_t, path_p, path_h, path_z = self.vsdn.forward(
            encoding, X, Mask, delta_t=delta_t, cov=cov, return_path=True)
        mean = path_p#.reshape(time_len, batch_size, -1, node_num, feature_size)
        mean = mean.transpose(0, 1)
        if return_h:
            return mean[:, 1:, :, :].mean(2), path_h, path_z, path_t
        else:
            return mean[:, 1:, :, :], loss, loss_recon, loss_kld

    def get_loss(self, data_batch):
        pred, loss, loss_recon, loss_kld = self.forward(data_batch=data_batch,
                                                        return_h=False)
        return loss, pred


class VSDN_ODE(nn.Module):
    def __init__(self,
                 enc_dimIn,
                 enc_dimHidden,
                 enc_numHidden,
                 enc_dimOut,
                 dim_rnn,
                 dim_state,
                 delta_t,
                 Lambda,
                 numSample,
                 enc_activation=nn.ReLU,
                 encoding='identity'):
        super().__init__()
        assert encoding in ['identity', 'dc', 'gc'], "Incorrect Encoder type."
        if encoding == 'gc':
            self.encoder = GCNet(enc_dimIn,
                                 enc_dimHidden,
                                 enc_numHidden,
                                 enc_dimOut,
                                 activation=enc_activation)
        elif encoding == 'dc':
            self.encoder = DCNet(enc_dimIn,
                                 enc_dimHidden,
                                 enc_numHidden,
                                 enc_dimOut,
                                 activation=enc_activation)
        else:
            self.encoder = Identity()#DensNet(enc_dimIn,enc_dimHidden,enc_numHidden,enc_dimOut,activation=enc_activation)#
        #self.decoder = DensNet(enc_dimOut,enc_dimHidden,enc_numHidden,enc_dimIn,activation=enc_activation)
        # TODO:
        self.vsdn = LatentODE(enc_dimOut,
                                    dim_state,
                                    dim_rnn,
                                    enc_dimIn,
                                    enc_dimHidden,
                                    enc_dimHidden,
                                    delta_t,
                                    None,
                                    default_num_samples=numSample)
        self.delta_t = delta_t
        self.numSample = numSample
        return

    def forward(self, data_batch, delta_t=None, numSample=None, return_h=True):
        if delta_t is None:
            delta_t = self.delta_t
        if numSample is None:
            numSample = self.numSample
        t = data_batch['t'][0]
        Mask = data_batch['masks']
        X = data_batch['values'] * Mask
        A_xt = data_batch['adjacent_matrices']
        #
        encoding = self.encoder(X, A_xt).transpose(0, 1)
        time_len, batch_size, node_num, feature_size = encoding.shape
        X = X.transpose(0, 1)#.reshape(time_len, batch_size * node_num, -1)
        Mask = Mask.transpose(0, 1)#.reshape(time_len, batch_size * node_num, -1)
        cov = torch.zeros(size=(*X.shape[1:-1], 1)).cuda()
        _, loss, path_t, path_p, path_z = self.vsdn.forward(
            encoding.reshape(time_len, batch_size * node_num, feature_size), Mask, delta_t=delta_t, cov=cov, return_path=True)
        mean = path_p#self.decoder(path_p.reshape(time_len, batch_size, -1, node_num, feature_size))
        #print(path_p.shape)
        mean = mean.transpose(0, 1)
        if return_h:
            return mean[:, 1:, :, :].mean(2), path_z, path_t
        else:
            return mean[:, 1:, :, :], loss

    def get_loss(self, data_batch):
        pred, loss = self.forward(data_batch=data_batch,
                                                        return_h=False)
        return loss, pred