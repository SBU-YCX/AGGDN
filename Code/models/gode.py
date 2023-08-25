"""
Author: Yingru Liu
GRU-ODE.
"""
import torch
import numpy as np
import torch.nn as nn
from models.modules import MaskedMSEloss
from models.modules import DCGRUCell, DCGRUODECell, GCGRUCell, GCGRUODECell, DensNet, DenseGRUODE, DDCGRUCell, DDCGRUODECell
from models.modules import GCNet as GCODECell
from models.modules import DCNet as DCODECell
from models.modules import DDCNet as DDCODECell
from models.modules import GRUCell, LSTMCell
from models.modules import IdentityNet as IDENTITYLayer


###############################################################################
#
#       Discrete-Time Models
#
###############################################################################

class FFNN(nn.Module):
    def __init__(self, dimIn, dimHidden, numHidden, numNode, learnstd=False):
        super(FFNN, self).__init__()
        self.Output = DensNet(dimIn, dimHidden, numHidden, dimIn)
        self.Output_std = DensNet(dimIn, dimHidden, numHidden, dimIn) if learnstd else None
        #self.Output = DensNet(dimIn * numNode, dimHidden * numNode, numHidden, dimIn * numNode)
        #self.Output_std = DensNet(dimIn * numNode, dimHidden * numNode, numHidden, dimIn * numNode) if learnstd else None
        self.criterion = MaskedMSEloss()
        return

    def forward(self, data_batch):
        X = data_batch['values'] * data_batch['masks']
        A_x = data_batch['adjacent_matrices']
        batchsize, lenSeq, numNode, dim = X.size()
        X_pred = self.Output(X)
        X_std = self.Output_std(
            X)[:, 0:-1, :, :] if self.Output_std is not None else None
        #X_pred = self.Output(X.reshape(batchsize, lenSeq, -1)).reshape(batchsize, lenSeq, numNode, -1)
        #X_std = self.Output_std(
        #    X.reshape(batchsize, lenSeq, -1)).reshape(batchsize, lenSeq, numNode, -1)[:, 0:-1, :, :] if self.Output_std is not None else None
        return X_pred[:, 0:-1, :, :], X_std

    def get_loss(self, data_batch):
        X_pred, X_std = self.forward(data_batch)
        X, M_v = data_batch['values'][:,
                                      1:, :, :], data_batch['masks'][:,
                                                                     1:, :, :]
        mse_v = self.criterion(X_pred, X,
                               M_v) if X_std is None else self.criterion(
                                   X_pred, X, M_v, X_std=X_std)
        return mse_v, X_pred

    def predict(self, X, A, delta_t=None, Delta_T=None, h=None, g=None, z=None):
        return None


'''
class LSTM(nn.Module):
    def __init__(self, dimIn, dimRnn, numRNNs, numNode, learnstd=False):
        super(LSTM, self).__init__()
        self.Encoder_V = nn.LSTM(dimIn, dimRnn, numRNNs * 2)
        #self.Encoder_V = nn.LSTM(dimIn * numNode, dimRnn * numNode, numRNNs * 2)
        self.Output = nn.Linear(dimRnn, dimIn)
        self.Output_std = nn.Linear(dimRnn, dimIn) if learnstd else None
        self.criterion = MaskedMSEloss()
        return

    def forward(self, data_batch):
        X = data_batch['values'] * data_batch['masks']
        A_x = data_batch['adjacent_matrices']
        batchsize, lenSeq, numNode, dim = X.size()
        Y = self.Encoder_V(X.permute(0, 2, 1, 3).reshape(-1, lenSeq, dim))[0].reshape(batchsize, numNode, lenSeq, -1).permute(0, 2, 1, 3)
        #Y = self.Encoder_V(X.reshape(batchsize, lenSeq, -1))[0].reshape(batchsize, lenSeq, numNode, -1)
        X_pred = self.Output(Y)
        X_std = self.Output_std(
            Y)[:, 0:-1, :, :] if self.Output_std is not None else None
        return X_pred[:, 0:-1, :, :], X_std

    def get_loss(self, data_batch):
        X_pred, X_std = self.forward(data_batch)
        X, M_v = data_batch['values'][:,
                                      1:, :, :], data_batch['masks'][:,
                                                                     1:, :, :]
        mse_v = self.criterion(X_pred, X,
                               M_v) if X_std is None else self.criterion(
                                   X_pred, X, M_v, X_std=X_std)
        return mse_v, X_pred

    def predict(self, X, A, delta_t=None, Delta_T=None, h=None, g=None, z=None):
        return None
'''


class LSTM(nn.Module):
    def __init__(self, dimIn, dimRnn, numRNNs, numNode, learnstd=False):
        super(LSTM, self).__init__()
        self.Encoder_V = nn.ModuleList()
        for block_idx in range(numRNNs):
            self.Encoder_V.append(
                LSTMCell(
                    dimIn if not block_idx else dimRnn,
                    dimRnn,
                ))
        #    self.Encoder_V.append(
        #        LSTMCell(
        #            dimIn * numNode if not block_idx else dimRnn * numNode,
        #            dimRnn * numNode,
        #        ))
        self.Output = nn.Linear(dimRnn, dimIn)
        self.Output_std = nn.Linear(dimRnn, dimIn) if learnstd else None
        self.criterion = MaskedMSEloss()
        return

    def forward(self, data_batch):
        X = data_batch['values'] * data_batch['masks']
        batchsize, lenSeq, numNode, dim = X.size()
        #X = X.reshape(batchsize, lenSeq, 1, -1)
        A_x = data_batch['adjacent_matrices']
        for block_v in self.Encoder_V:
            X_new = []
            hv, cv = block_v.initiation()
            hv = hv.repeat(X.size(0), X.size(2), 1)
            cv = cv.repeat(X.size(0), X.size(2), 1)
            for n in range(X.size(1)):
                hv, cv = block_v(X[:, n, :, :], hv, cv, A_x)
                X_new.append(hv)
            X = torch.stack(X_new, 1)
        #X = X.reshape(batchsize, lenSeq, numNode, -1)
        X_pred = self.Output(X)
        X_std = self.Output_std(
            X)[:, 0:-1, :, :] if self.Output_std is not None else None
        return X_pred[:, 0:-1, :, :], X_std

    def get_loss(self, data_batch):
        X_pred, X_std = self.forward(data_batch)
        X, M_v = data_batch['values'][:,
                                      1:, :, :], data_batch['masks'][:,
                                                                     1:, :, :]
        mse_v = self.criterion(X_pred, X,
                               M_v) if X_std is None else self.criterion(
                                   X_pred, X, M_v, X_std=X_std)
        return mse_v, X_pred

    def predict(self, X, A, delta_t=None, Delta_T=None, h=None, g=None, z=None):
        return None


class GRU(nn.Module):
    def __init__(self, dimIn, dimRnn, numRNNs, numNode, learnstd=False):
        super(GRU, self).__init__()
        self.Encoder_V = nn.ModuleList()
        for block_idx in range(numRNNs):
            self.Encoder_V.append(
                GRUCell(
                    dimIn if not block_idx else dimRnn,
                    dimRnn,
                ))
        #    self.Encoder_V.append(
        #        GRUCell(
        #            dimIn * numNode if not block_idx else dimRnn * numNode,
        #            dimRnn * numNode,
        #        ))
        self.Output = nn.Linear(dimRnn, dimIn)
        self.Output_std = nn.Linear(dimRnn, dimIn) if learnstd else None
        self.criterion = MaskedMSEloss()
        return

    def forward(self, data_batch):
        X = data_batch['values'] * data_batch['masks']
        batchsize, lenSeq, numNode, dim = X.size()
        #X = X.reshape(batchsize, lenSeq, 1, -1)
        A_x = data_batch['adjacent_matrices']
        for block_v in self.Encoder_V:
            X_new = []
            hv = block_v.initiation().repeat(X.size(0), X.size(2), 1)
            for n in range(X.size(1)):
                hv = block_v(X[:, n, :, :], hv, A_x)
                X_new.append(hv)
            X = torch.stack(X_new, 1)
        #X = X.reshape(batchsize, lenSeq, numNode, -1)
        X_pred = self.Output(X)
        X_std = self.Output_std(
            X)[:, 0:-1, :, :] if self.Output_std is not None else None
        return X_pred[:, 0:-1, :, :], X_std

    def get_loss(self, data_batch):
        X_pred, X_std = self.forward(data_batch)
        X, M_v = data_batch['values'][:,
                                      1:, :, :], data_batch['masks'][:,
                                                                     1:, :, :]
        mse_v = self.criterion(X_pred, X,
                               M_v) if X_std is None else self.criterion(
                                   X_pred, X, M_v, X_std=X_std)
        return mse_v, X_pred

    def predict(self, X, A, delta_t=None, Delta_T=None, h=None, g=None, z=None):
        return None


class GCGRU(nn.Module):
    def __init__(self, dimIn, dimRnn, numRNNs, learnstd=False):
        super(GCGRU, self).__init__()
        self.Encoder_V = nn.ModuleList()
        for block_idx in range(numRNNs):
            self.Encoder_V.append(
                GCGRUCell(
                    dimIn if not block_idx else dimRnn,
                    dimRnn,
                ))
        self.Output = nn.Linear(dimRnn, dimIn)
        self.Output_std = nn.Linear(dimRnn, dimIn) if learnstd else None
        self.criterion = MaskedMSEloss()
        return

    def forward(self, data_batch):
        X = data_batch['values'] * data_batch['masks']
        A_x = data_batch['adjacent_matrices']
        for block_v in self.Encoder_V:
            X_new = []
            hv = block_v.initiation().repeat(X.size(0), X.size(2), 1)
            for n in range(X.size(1)):
                hv = block_v(X[:, n, :, :], hv, A_x)
                X_new.append(hv)
            X = torch.stack(X_new, 1)
        X_pred = self.Output(X)
        X_std = self.Output_std(
            X)[:, 0:-1, :, :] if self.Output_std is not None else None
        return X_pred[:, 0:-1, :, :], X_std

    def get_loss(self, data_batch):
        X_pred, X_std = self.forward(data_batch)
        X, M_v = data_batch['values'][:,
                                      1:, :, :], data_batch['masks'][:,
                                                                     1:, :, :]
        mse_v = self.criterion(X_pred, X,
                               M_v) if X_std is None else self.criterion(
                                   X_pred, X, M_v, X_std=X_std)
        return mse_v, X_pred

    def predict(self, X, A, delta_t=None, Delta_T=None, h=None, g=None, z=None):
        for block_v in self.Encoder_V:
            if h is None:
                h = block_v.initiation().repeat(X.size(0), X.size(2), 1)
            X = block_v(X, h, A)
        X_pred = self.Output(X)
        X_std = self.Output_std(X) if self.Output_std is not None else None
        return X_pred, X_std, h, g, z


class DCGRU(GCGRU):
    def __init__(self, dimIn, dimRnn, numRNNs, K, learnstd=False):
        super(DCGRU, self).__init__(dimIn, dimRnn, numRNNs, learnstd=learnstd)
        self.Encoder_V = nn.ModuleList()
        for block_idx in range(numRNNs):
            self.Encoder_V.append(
                DCGRUCell(dimIn if not block_idx else dimRnn, dimRnn, K))
        return


class DDCGRU(GCGRU):
    def __init__(self, dimIn, dimRnn, numRNNs, numNode, K, learnstd=False):
        super(DDCGRU, self).__init__(dimIn, dimRnn, numRNNs, learnstd=learnstd)
        self.Encoder_V = nn.ModuleList()
        for block_idx in range(numRNNs):
            self.Encoder_V.append(
                DDCGRUCell(dimIn if not block_idx else dimRnn, dimRnn, K, numNode=numNode))
        return

'''
class GRU(nn.Module):
    def __init__(self, dimIn, dimRnn, numRNNs, numNode, learnstd=False):
        super(GRU, self).__init__()
        self.Encoder_V = nn.GRU(dimIn, dimRnn, numRNNs)
        #self.Encoder_V = nn.GRU(dimIn * numNode, dimRnn * numNode, numRNNs)
        self.Output = nn.Linear(dimRnn, dimIn)
        self.Output_std = nn.Linear(dimRnn, dimIn) if learnstd else None
        self.criterion = MaskedMSEloss()
        return

    def forward(self, data_batch):
        X = data_batch['values'] * data_batch['masks']
        A_x = data_batch['adjacent_matrices']
        batchsize, lenSeq, numNode, dim = X.size()
        Y = self.Encoder_V(X.permute(0, 2, 1, 3).reshape(-1, lenSeq, dim))[0].reshape(batchsize, numNode, lenSeq, -1).permute(0, 2, 1, 3)
        #Y = self.Encoder_V(X.reshape(batchsize, lenSeq, -1))[0].reshape(batchsize, lenSeq, numNode, -1)
        X_pred = self.Output(Y)
        X_std = self.Output_std(
            Y)[:, 0:-1, :, :] if self.Output_std is not None else None
        return X_pred[:, 0:-1, :, :], X_std

    def get_loss(self, data_batch):
        X_pred, X_std = self.forward(data_batch)
        X, M_v = data_batch['values'][:,
                                      1:, :, :], data_batch['masks'][:,
                                                                     1:, :, :]
        mse_v = self.criterion(X_pred, X,
                               M_v) if X_std is None else self.criterion(
                                   X_pred, X, M_v, X_std=X_std)
        return mse_v, X_pred

    def predict(self, X, A, delta_t=None, Delta_T=None, h=None, g=None, z=None):
        return None
'''


###############################################################################
#
#       Continuous-Time Models
#
###############################################################################

class NODE(nn.Module):
    def __init__(self,
                 dimIn,
                 dimRnn, 
                 dimState,
                 dimODEHidden,
                 numODEHidden,
                 delta_t,
                 beta=1.0,
                 K=3,
                 learnstd=False, 
                 ode_method = 'euler'):
        super(NODE, self).__init__()
        # ODE Part.
        self.GRU = GRUCell(dimIn, dimRnn)
        self.ODE = DensNet(dimRnn, dimODEHidden, numODEHidden, dimRnn)
        self.Output = nn.Linear(dimRnn, dimIn)
        self.Output_std = nn.Linear(dimRnn, dimIn) if learnstd else None
        #
        self.beta = beta
        self.delta_t = delta_t
        self.criterion = MaskedMSEloss()
        self.iteration = 0
        # initial state.
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dimRnn)))
        return

    def forward(self, data_batch, delta_t = None, return_h = True):
        if delta_t is None:
            delta_t = self.delta_t
        t = data_batch['t'][0]
        Mask = data_batch['masks']
        X = data_batch['values'] * Mask
        A_xt = data_batch['adjacent_matrices']
        #
        t_current, t_max, n = 0., t[-1], 0
        hv = self.h0.repeat(X.size(0), X.size(2), 1)
        hv_pre, hv_pre_N, hv_post_N = [], [], []
        traj_t = []
        """ODE Part."""
        while t_current <= t_max + 1e-4 * delta_t:
            # update hv from .
            dhv_pre = self.ODE(hv, A_xt)
            hv1 = hv + dhv_pre * delta_t
            hv_pre.append(hv1)
            # reach an observation frame.
            if t_current > t[n] - 1e-4 * delta_t:
                # save the prior feature and state.
                hv_pre_N.append(hv1)
                # update the posterior state and feature.
                hv2 = self.GRU(X[:, n, :, :], hv1, A_xt)
                obs_mask = (Mask[:, n, :, :].abs().sum(
                    (-1), keepdims=True) > 1e-4).type(
                        torch.cuda.FloatTensor)
                hv = hv1 * (1. - obs_mask) + hv2 * obs_mask
                hv_post_N.append(hv)
            else:
                hv = hv1
            # update time.
            traj_t.append(t_current)
            if t_current > t[n] - 1e-4 * delta_t:
                t_current = t[n].item()
                n += 1
            t_current += delta_t
        # stack the features.
        hv_pre = torch.stack(hv_pre, 1)
        hv_pre_N = torch.stack(hv_pre_N, 1)
        hv_post_N = torch.stack(hv_post_N, 1)
        traj_t = torch.Tensor(traj_t).cuda()
        #
        X_pred = self.Output(hv_pre_N)
        X_std = self.Output_std(
            hv_pre_N)[:, 1:, :, :] if self.Output_std is not None else None
        if return_h:
            return X_pred[:,
                          1:, :, :], X_std, hv_pre, hv_pre_N, hv_post_N, traj_t
        else:
            return X_pred[:, 1:, :, :], X_std
        

    def get_loss(self, data_batch):
        X_pred, X_std = self.forward(data_batch, self.delta_t,
                                     return_h=True)[0:2]
        X, M_v = data_batch['values'][:,
                                      1:, :, :], data_batch['masks'][:,
                                                                     1:, :, :]
        mse_v = self.criterion(X_pred, X,
                               M_v) if X_std is None else self.criterion(
                                   X_pred, X, M_v, X_std=X_std)
        return mse_v, X_pred


class GCODERNN(NODE):
    def __init__(self, dimIn, dimRnn, dimState, dimODEHidden, numODEHidden, delta_t,
                 beta=1.0, K=3, learnstd=False, ode_method = 'euler'):
        super(GCODERNN, self).__init__(dimIn,
                                       dimRnn,
                                       dimState,
                                       dimODEHidden,
                                       numODEHidden,
                                       delta_t,
                                       beta,
                                       K,
                                       learnstd, 
                                       ode_method)
        self.GRU = GCGRUCell(dimIn, dimRnn)
        self.ODE = GCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn)
        return


class DCODERNN(NODE):
    def __init__(self, dimIn, dimRnn, dimState, dimODEHidden, numODEHidden, delta_t,
                 beta=1.0, K=3, learnstd=False, ode_method = 'euler'):
        super(DCODERNN, self).__init__(dimIn,
                                       dimRnn,
                                       dimState,
                                       dimODEHidden,
                                       numODEHidden,
                                       delta_t,
                                       beta,
                                       K,
                                       learnstd, 
                                       ode_method)
        self.GRU = DCGRUCell(dimIn, dimRnn, K=K)
        self.ODE = DCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn, K=K)
        return


class DDCODERNN(NODE):
    def __init__(self, dimIn, dimRnn, dimState, dimODEHidden, numODEHidden, numNode, delta_t,
                 beta=1.0, K=3, learnstd=False, ode_method = 'euler'):
        super(DDCODERNN, self).__init__(dimIn,
                                        dimRnn,
                                        dimState,
                                        dimODEHidden,
                                        numODEHidden,
                                        delta_t,
                                        beta,
                                        K,
                                        learnstd, 
                                        ode_method)
        self.GRU = DDCGRUCell(dimIn, dimRnn, K=K, numNode=numNode)
        self.ODE = DDCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn, K=K, numNode=numNode)
        return


class GCGRUODE(GCODERNN):
    def __init__(self, dimIn, dimRnn, dimState, dimODEHidden, numODEHidden, delta_t,
                 beta=1.0, K=3, learnstd=False, ode_method = 'euler'):
        super(GCGRUODE, self).__init__(dimIn,
                                       dimRnn,
                                       dimState,
                                       1,
                                       1,
                                       delta_t,
                                       beta,
                                       K,
                                       learnstd, 
                                       ode_method)
        # change the ODE component to GRU cell.
        self.ODE = GCGRUODECell(dimRnn)


class DCGRUODE(DCODERNN):
    def __init__(self, dimIn, dimRnn, dimState, dimODEHidden, numODEHidden, delta_t,
                 beta=1.0, K=3, learnstd=False, ode_method = 'euler'):
        super(DCGRUODE, self).__init__(dimIn,
                                       dimRnn,
                                       dimState,
                                       1,
                                       1,
                                       delta_t,
                                       beta,
                                       K,
                                       learnstd, 
                                       ode_method)
        self.ODE = DCGRUODECell(dimRnn, K=K)
        return


class DDCGRUODE(DCODERNN):
    def __init__(self, dimIn, dimRnn, dimState, dimODEHidden, numODEHidden, numNode, delta_t,
                 beta=1.0, K=3, learnstd=False, ode_method = 'euler'):
        super(DDCGRUODE, self).__init__(dimIn,
                                        dimRnn,
                                        dimState,
                                        1,
                                        1,
                                        delta_t,
                                        beta,
                                        K,
                                        learnstd, 
                                        ode_method)
        self.ODE = DDCGRUODECell(dimRnn, K=K, numNode=numNode)
        return


class HGDCODE(nn.Module):
    def __init__(self,
                 dimIn,
                 dimRnn,
                 dimState,
                 dimODEHidden,
                 numODEHidden,
                 numNode, 
                 delta_t,
                 beta=1.0,
                 K=3,
                 learnstd=False,
                 ode_method = 'euler'):
        super(HGDCODE, self).__init__()
        # ODE Part.
        self.GRU_V = DDCGRUCell(dimIn, dimRnn, K=K, numNode=numNode)
        self.ODE_V = DDCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn, K=K, numNode=numNode)
        self.GRU_G = DDCGRUCell(dimIn, dimRnn, K=K, numNode=numNode)
        self.ODE_G = DDCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn, K=K, numNode=numNode)
        self.Net_Zout = nn.Sequential(nn.Linear(dimRnn, dimRnn), nn.Tanh(),
                                      nn.Linear(dimRnn, dimRnn), nn.Tanh(),
                                      nn.Linear(dimRnn, dimRnn), nn.Sigmoid())
        self.Output = nn.Linear(dimRnn, dimIn)
        self.Output_std = nn.Linear(dimRnn, dimIn) if learnstd else None
        self.Similarity = nn.TripletMarginLoss()
        #
        self.beta = beta
        self.delta_t = delta_t
        self.criterion = MaskedMSEloss()
        self.iteration = 0
        # initial state.
        self.z0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dimRnn)))
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dimRnn)))
        return

    def forward(self, data_batch, delta_t=None, return_h=True):
        if delta_t is None:
            delta_t = self.delta_t
        t = data_batch['t'][0]
        Mask = data_batch['masks']
        X = data_batch['values'] * Mask
        A_xt = data_batch['adjacent_matrices']
        #
        t_current, t_max, n = 0., t[-1], 0
        hv = self.h0.repeat(X.size(0), X.size(2), 1)
        zv = self.z0.repeat(X.size(0), X.size(2), 1)
        hv_pre, hv_pre_N, hv_post_N = [], [], []
        zv_pre, zv_pre_N, zv_post_N = [], [], []
        traj_t = []
        """ODE Part."""
        while t_current <= t_max + 1e-4 * delta_t:
            # update hv from .
            dhv_pre, dzv_pre = self.ODE_V(hv, A_xt), self.ODE_G(zv, A_xt)
            hv1 = hv + dhv_pre * delta_t
            zv1 = zv + dzv_pre * delta_t
            hv_pre.append(hv1)
            zv_pre.append(zv1)
            # reach an observation frame.
            if t_current > t[n] - 1e-4 * delta_t:
                # save the prior feature and state.
                hv_pre_N.append(hv1)
                zv_pre_N.append(zv1)
                # update the posterior state and feature.
                hv2 = self.GRU_V(X[:, n, :, :], hv1, A_xt)
                zv2 = self.GRU_G(X[:, n, :, :], zv1, A_xt)
                obs_mask = (Mask[:, n, :, :].abs().sum(
                    (-1), keepdims=True) > 1e-4).type(
                        torch.cuda.FloatTensor)
                hv = hv1 * (1. - obs_mask) + hv2 * obs_mask
                zv = zv1 * (1. - obs_mask) + zv2 * obs_mask
                hv_post_N.append(hv)
                zv_post_N.append(zv)
            else:
                hv, zv = hv1, zv1
            # update time.
            traj_t.append(t_current)
            if t_current > t[n] - 1e-4 * delta_t:
                t_current = t[n].item()
                n += 1
            t_current += delta_t
        # stack the features.
        hv_pre = torch.stack(hv_pre, 1)
        hv_pre_N = torch.stack(hv_pre_N, 1)
        hv_post_N = torch.stack(hv_post_N, 1)
        zv_pre = torch.stack(zv_pre, 1)
        zv_pre_N = torch.stack(zv_pre_N, 1)
        zv_post_N = torch.stack(zv_post_N, 1)
        traj_t = torch.Tensor(traj_t).cuda()
        #
        gate = self.Net_Zout(zv_pre_N)
        #gate1 = self.Net_Zout(zv_pre)
        X_pred = self.Output(hv_pre_N * gate)#self.Output(hv_pre * gate1)#
        X_std = self.Output_std(
            hv_pre_N * gate)[:,
                             1:, :, :] if self.Output_std is not None else None
        if return_h:
            return X_pred[:,
                          1:, :, :], X_std, hv_pre, hv_pre_N * gate, hv_post_N * gate, traj_t
        else:
            return X_pred[:, 1:, :, :], X_std

    def get_loss(self, data_batch):
        X_pred, X_std = self.forward(data_batch, self.delta_t,
                                     return_h=True)[0:2]
        X, M_v = data_batch['values'][:,
                                      1:, :, :], data_batch['masks'][:,
                                                                     1:, :, :]
        mse_v = self.criterion(X_pred, X,
                               M_v) if X_std is None else self.criterion(
                                   X_pred, X, M_v, X_std=X_std)
        return mse_v, X_pred

    def predict(self, X, A, delta_t, Delta_T, h=None, g=None, z=None, return_h=False):
        if X is not None:
            if h is None:
                h = self.h0.repeat(X.size(0), X.size(1), 1)
            if g is None:
                g = self.z0.repeat(X.size(0), X.size(1), 1)
            # encode G
            hv2 = self.GRU_V(X, h, A)
            gv2 = self.GRU_G(X, g, A)
            # combine
            obs_mask = (X.abs().sum((-1), keepdims=True) > 1e-4).type(torch.cuda.FloatTensor)
            h = h * (1. - obs_mask) + hv2 * obs_mask
            g = g * (1. - obs_mask) + gv2 * obs_mask
        # propagate F
        t_cur = 0.
        while t_cur <= Delta_T + 1e-4 * delta_t:
            dhv_pre = self.ODE_V(h, A)
            dgv_pre = self.ODE_G(g, A)
            h = h + dhv_pre * delta_t
            g = g + dgv_pre * delta_t
            t_cur += delta_t
        gate = self.Net_Zout(g)
        X_pred = self.Output(h * gate)
        X_std = self.Output_std(h * gate) if self.Output_std is not None else None
        if return_h: 
            return X_pred, X_std, h * gate, h, g
        else:
            return X_pred, X_std, h, g, z


class NGDCSDE(nn.Module):
    def __init__(self,
                 dimIn,
                 dimRnn,
                 dimState,
                 numRNNs,
                 dimODEHidden,
                 numODEHidden,
                 numNode, 
                 delta_t,
                 beta=1.0,
                 numSample=5,
                 K=3,
                 learnstd=False, 
                 ode_method = 'euler', 
                 sde_method = 'euler', 
                 connect_method = 'r'):
        super(NGDCSDE, self).__init__()
        # ODE Part.
        self.ODE = HGDCODE(dimIn,
                           dimRnn,
                           dimState,
                           dimODEHidden,
                           numODEHidden,
                           numNode, 
                           delta_t,
                           K=K,
                           beta=beta, 
                           ode_method = ode_method)
        # SDE Part.
        #self.Encoder_post = nn.GRU(dimRnn, dimState, bidirectional=True)                ## [20220113] modified (dimRnn -> dimState)
        #self.Encoder_post = DenseGRUODE(dimRnn, dimState, delta_t)                       ## [20220116] added
        self.Encoder_pre = nn.GRU(dimRnn, dimRnn, bidirectional = False)
        self.Encoder_post = nn.GRU(dimRnn, dimRnn, bidirectional = True) ## [20220117] added
        self.Net_diff = DensNet(dimRnn, dimODEHidden, numODEHidden, dimState)
        self.Net_drift = DensNet(dimState, dimODEHidden, numODEHidden, dimState)
        self.Net_prior = DensNet(dimRnn, dimODEHidden,
                                numODEHidden, dimState)
        self.Net_post = DensNet(dimRnn * 2 + dimState, dimODEHidden,
                                numODEHidden, dimState)
        self.Output = nn.Linear(dimRnn + dimState, dimIn)                           ## [20220113] modified (dimRnn + dimState -> dimRnn + 2 * dimState)
        ##self.Output = nn.Linear(dimState, dimIn) #IDENTITYLayer()
        self.Output_std = nn.Linear(dimRnn + 
                                    dimState, dimIn) if learnstd else None
        #self.Output_std = nn.Linear(dimState, dimIn) if learnstd else None #IDENTITYLayer() if learnstd else None
        #
        #self.Net_res = DensNet(dimRnn, dimODEHidden, numODEHidden, dimState)           ## [20220114] added 
        self.beta = beta
        self.delta_t = delta_t
        self.criterion = MaskedMSEloss()
        self.numSample = numSample
        # initial state.
        self.z0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dimState)))
        return

    def forward(self, data_batch, delta_t=None, numSample=None, return_h=True):
        if delta_t is None:
            delta_t = self.delta_t
        if numSample is None:
            numSample = self.numSample
        #
        ode_pred, _, hv_pre, hv_pre_N, hv_post_N, traj_t = self.ODE.forward(
            data_batch, delta_t, return_h=True)
        # compute the inference model.
        batchsize, lenSeq, numNode, dim = hv_pre_N.size()
        #batchsize, lenSeq1, numNode, dim = hv_pre.size()
        #Z_embed_pre = self.Encoder_pre(                                                   ## [20220116] added
        #    hv_pre_N.transpose(0, 1).reshape(lenSeq1, batchsize * numNode,
        #                                   dim))[0]
        #Z_embed_pre = Z_embed_pre.reshape(lenSeq1, batchsize, numNode,                          ## [20220114] removed
        #                          Z_embed_pre.size(-1)).transpose(0, 1)
        #Z_embed_pre = hv_pre_N
        #batchsize, lenSeq2, numNode, dim = hv_post_N.size()
        #Z_embed_post = self.Encoder_post(                                                   
        #    hv_post_N.transpose(0, 1).reshape(lenSeq2, batchsize * numNode,
        #                                   dim))[0]
        #Z_embed_post = Z_embed_post.reshape(lenSeq2, batchsize, numNode,
        #                          Z_embed_post.size(-1)).transpose(0, 1)
        #Z_embed_post = hv_post_N
        
        Z_embed = self.Encoder_post(                                                   ## [20220116] added
            hv_pre_N.transpose(0, 1).reshape(lenSeq, batchsize * numNode,
                                           dim))[0]
        Z_embed = Z_embed.reshape(lenSeq, batchsize, numNode,                          ## [20220114] removed
                                  Z_embed.size(-1)).transpose(0, 1)
        Z_embed_post = Z_embed
        Z_embed_pre = torch.chunk(Z_embed, 2, -1)[0]
        
        kld = 0.
        #
        Z = self.z0.unsqueeze(0).repeat(numSample, Z_embed_pre.size(0),
                                        Z_embed_pre.size(2), 1)
        t, n_frame = data_batch['t'][0], 0                                             ## [20220114] removed
        #Zs, Zs_N, Z_priors = [], [], []                                                ## [20220114] removed    
        Zs_N = []
        for n, t_n in enumerate(traj_t):                                               ## [20220114] removed
            #
            if n_frame < t.size(0) and t_n > t[n_frame] - 1e-4 * delta_t:  
                prior = torch.tanh(self.Net_prior(Z_embed_pre[:, n_frame, :, :]))
                diff = self.Net_diff(Z_embed_pre[:, n_frame, :, :])#0.1 * torch.tanh()
            Z_diff = Z * diff
            Z_prior = 0.5 * prior * (Z_diff ** 2)#torch.tanh(self.Net_drift(prior * Z))#self.Net_drift(prior * ((Z_diff * Z) ** 2))) #self.Net_drift(prior)
            #prior = self.Net_prior(torch.cat([Z_embed_pre[:, n, :, :].unsqueeze(0).repeat(
            #                numSample, 1, 1, 1), Z], dim=-1))
            #Z_prior = torch.tanh(self.Net_drift(prior))
            #Z_diff = self.Net_diff(torch.cat([Z_embed_pre[:, n, :, :].unsqueeze(0).repeat(
            #                numSample, 1, 1, 1), Z], dim=-1))
            if n_frame < t.size(0) and t_n > t[n_frame] - 1e-4 * delta_t and not return_h: 
                Z_post = self.Net_post(torch.cat([Z_embed_post[:, n_frame, :, :].unsqueeze(0).repeat(
                            numSample, 1, 1, 1), Z], dim=-1))#Z_embed_post[:, n_frame, :, :])#
                #Z_post = torch.tanh(self.Net_drift(post * Z))
                Z = Z + Z_post * delta_t + np.sqrt(
                    delta_t) * Z_diff * torch.randn_like(Z_diff)
                kld_t = (Z_post - Z_prior)**2 / torch.max(
                    Z_diff**2,
                    torch.Tensor([1e-2]).cuda())
                kld = kld + delta_t * kld_t.sum((-3, -2, -1)).mean()
            else: 
                Z = Z + Z_prior * delta_t + np.sqrt(
                    delta_t) * Z_diff * torch.randn_like(Z_diff)
            #Zs.append(Z)
            if n_frame < t.size(0) and t_n > t[n_frame] - 1e-4 * delta_t:                       ## [20220114] removed
                Zs_N.append(Z)# / (Z.max(-1, keepdim=True)[0]))
        #        Zs_N.append(Z_embed[:, n, :, :])#.unsqueeze(0).repeat(numSample, 1, 1, 1))        ## [20220113] added
                n_frame += 1                                                                    ## [20220114] removed
        #
        #Zs = torch.stack(Zs, -3)
        Zs_N = torch.stack(Zs_N, -3)                                                            ## [20220114] removed
        #Z_priors = torch.stack(Z_priors, -3)
        #
        #print(hv_pre_N.size(), Zs_N.size())
        #Zsp_N = self.Net_res(hv_pre_N)                                                            ## [20220114] added
        inFeature = torch.cat(
            [hv_pre_N.unsqueeze(0).repeat(numSample, 1, 1, 1, 1), Zs_N], -1)                     ## [20220114] modified (hv_pre_N.unsqueeze(0).repeat(numSample, 1, 1, 1, 1) -> hv_pre_N)
        #inFeature = Zs_N
        X_pred_residual = self.Output(inFeature)
        X_std = self.Output_std(
            inFeature)[:, :, 1:, :, :] if self.Output_std is not None else None
        X_pred = X_pred_residual[:, :, 1:, :, :] + ode_pred.unsqueeze(0)                                                
        if return_h:
            return X_pred.mean(0), hv_pre, hv_pre_N, hv_post_N, Zs_N             
        else:
            return X_pred, X_std, kld, Zs_N

    def get_loss(self, data_batch):
        X_pred, X_std, kld = self.forward(data_batch,
                                          self.delta_t,
                                          return_h=False)[0:3]
        #X = data_batch['ori_values'][:, 1:, :, :].unsqueeze(0)
        X = data_batch['values'][:, 1:, :, :].unsqueeze(0)
        M_v = data_batch['masks'][:, 1:, :, :].unsqueeze(0)
        mse_v = self.criterion(X_pred, X,
                               M_v) if X_std is None else self.criterion(
                                   X_pred, X, M_v, X_std=X_std)
        print(mse_v.detach().cpu().numpy(), '\t', kld.detach().cpu().numpy(), '\t', self.beta)
        return mse_v + self.beta * kld, X_pred.mean(0) # 

    def predict(self, X, A, delta_t, Delta_T, h=None, g=None, z=None, numSample=5): 
        ode_pred, _, hv_N, h, g = self.ODE.predict(X, A, delta_t, Delta_T, h, g, True)
        batchsize, numNode, dim = hv_N.size()
        Z_pre = self.Encoder_pre(hv_N.reshape(1, batchsize * numNode, dim))[0]
        Z_pre = Z_pre.reshape(batchsize, numNode, Z_pre.size(-1))
        if z is None:
            z = self.z0.unsqueeze(0).repeat(numSample, Z_pre.size(0), Z_pre.size(1), 1)
        t_cur = 0.
        prior = self.Net_prior(Z_pre)
        Z_diff = self.Net_diff(Z_pre)
        while t_cur <= Delta_T + 1e-4 * delta_t:
            Z_prior = self.Net_drift(prior * ((Z_diff * z) ** 2))
            z = z + Z_prior * delta_t + Z_diff * np.sqrt(delta_t) * torch.randn_like(Z_diff)
            t_cur += delta_t
        Z_N = z / z.max(-1, keepdim=True)[0]
        inFeature = torch.cat([hv_N.unsqueeze(0).repeat(numSample, 1, 1, 1), Z_N], -1)
        X_pred_residual = self.Output(inFeature)
        X_std = self.Output_std(inFeature) if self.Output_std is not None else None
        X_pred = X_pred_residual + ode_pred.unsqueeze(0) 
        return X_pred.mean(0), X_std, h, g, z 



#    diff_coef, scale = diffs[:, n, :, :], scales[:, n, :, :]
        #    diff = Z * diff_coef
        #    prior = 0.5 * scale * (diff**2)
            # reach an observation.
        #    post = self.Net_post(
        #        torch.cat([
        #            Z_embed[:, n, :, :].unsqueeze(0).repeat(
        #                numSample, 1, 1, 1), Z
        #        ], -1))
        #    Z_priors.append(prior)
        #    if not return_h:  # training
        #        Z = Z + post * delta_t + np.sqrt(
        #            delta_t) * diff * torch.randn_like(diff)
        #        kld_t = (post - prior)**2 / torch.max(
        #            diff**2,
        #            torch.Tensor([1e-2]).cuda())
        #        kld = kld + delta_t * kld_t.sum((-3, -2, -1)).mean()
        #    else:
        #        Z = Z + prior * delta_t + np.sqrt(
        #            delta_t) * diff * torch.randn_like(diff)
        #    Zs.append(Z)