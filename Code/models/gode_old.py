"""
Author: Yingru Liu
GRU-ODE.
"""
import torch
import numpy as np
import torch.nn as nn
from models.modules import MaskedMSEloss
from models.modules import DCGRUCell, DCGRUODECell, GCGRUCell, GCGRUODECell, DensNet, DenseGRUODE
from models.modules import GCNet as GCODECell
from models.modules import DCNet as DCODECell
from models.modules import IdentityNet as IDENTITYLayer


class GCODERNN(nn.Module):
    def __init__(self,
                 dimIn,
                 dimRnn,
                 numRNNs,
                 dimODEHidden,
                 numODEHidden,
                 delta_t,
                 learnstd=False):
        super(GCODERNN, self).__init__()
        self.GRU_V = GCGRUCell(dimIn, dimRnn)
        self.ODE_V = GCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn)
        self.Output = nn.Linear(dimRnn, dimIn)
        self.Output_std = nn.Linear(dimRnn, dimIn) if learnstd else None
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dimRnn)))
        self.criterion = MaskedMSEloss()
        self.delta_t = delta_t
        return

    def forward(self, data_batch, delta_t=None, return_h=False):
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
            dhv_pre = self.ODE_V(hv, A_xt)
            hv1 = hv + dhv_pre * delta_t
            hv_pre.append(hv1)
            # reach an observation frame.
            if t_current > t[n] - 1e-4 * delta_t:
                # save the prior feature and state.
                hv_pre_N.append(hv1)
                # update the posterior state and feature.
                hv2 = self.GRU_V(X[:, n, :, :], hv1, A_xt)
                obs_mask = (Mask[:, n, :, :].abs().sum(
                    (-1, -2), keepdims=True) > 1e-4).type(
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
        X_pred, X_std = self.forward(data_batch, self.delta_t)
        X, M_v = data_batch['values'][:,
                                      1:, :, :], data_batch['masks'][:,
                                                                     1:, :, :]
        mse_v = self.criterion(X_pred, X,
                               M_v) if X_std is None else self.criterion(
                                   X_pred, X, M_v, X_std=X_std)
        return mse_v, X_pred


class DCODERNN(GCODERNN):
    def __init__(self,
                 dimIn,
                 dimRnn,
                 numRNNs,
                 dimODEHidden,
                 numODEHidden,
                 delta_t,
                 K=3,
                 learnstd=False):
        super(DCODERNN, self).__init__(dimIn,
                                       dimRnn,
                                       numRNNs,
                                       dimODEHidden,
                                       numODEHidden,
                                       delta_t,
                                       learnstd=learnstd)
        self.GRU_V = DCGRUCell(dimIn, dimRnn, K=K)
        self.ODE_V = DCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn, K=K)
        return


class GCGRUODE(GCODERNN):
    def __init__(self, dimIn, dimRnn, numRNNs, delta_t, learnstd=False):
        super(GCGRUODE, self).__init__(dimIn,
                                       dimRnn,
                                       numRNNs,
                                       1,
                                       1,
                                       delta_t,
                                       learnstd=learnstd)
        # change the ODE component to GRU cell.
        self.ODE_V = GCGRUODECell(dimRnn)


class DCGRUODE(DCODERNN):
    def __init__(self, dimIn, dimRnn, numRNNs, delta_t, K=3, learnstd=False):
        super(DCGRUODE, self).__init__(dimIn,
                                       dimRnn,
                                       numRNNs,
                                       1,
                                       1,
                                       delta_t,
                                       K=K,
                                       learnstd=learnstd)
        self.ODE_V = DCGRUODECell(dimRnn, K=K)
        return


class HGDCODE(nn.Module):
    def __init__(self,
                 dimIn,
                 dimRnn,
                 dimState,
                 dimODEHidden,
                 numODEHidden,
                 delta_t,
                 beta=1.0,
                 K=3,
                 learnstd=False):
        super(HGDCODE, self).__init__()
        # ODE Part.
        self.GRU_V = DCGRUCell(dimIn, dimRnn, K=K)
        self.ODE_V = DCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn, K=K)
        self.GRU_G = DCGRUCell(dimIn, dimRnn, K=K)
        self.ODE_G = DCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn, K=K)
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
                    (-1, -2), keepdims=True) > 1e-4).type(
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
        X_pred = self.Output(hv_pre_N * gate)
        X_std = self.Output_std(
            hv_pre_N * gate)[:,
                             1:, :, :] if self.Output_std is not None else None
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


class DenseODE(nn.Module):
    def __init__(self, dimRnn, dimState, dimODEHidden, numODEHidden, delta_t):
        super(DenseODE, self).__init__()
        self.NN = DensNet(dimRnn + dimState, dimODEHidden, numODEHidden, dimState)
        self.delta_t = delta_t
        self.h0 = nn.Parameter(0.01 * torch.randn(size = (1, dimState)))

    def forward(self, X, delta_t = None):
        if not delta_t:
            delta_t = self.delta_t
        T = X.size(0)
        h = self.h0.repeat(X.size(1), 1)
        hs = []
        for t in range(T):
            h = h + self.NN(torch.cat([X[t], h], -1)) * delta_t
            hs.append(h)
        hs = torch.stack(hs, 1)
        return hs


class NGDCSDE(nn.Module):
    def __init__(self,
                 dimIn,
                 dimRnn,
                 dimState,
                 numRNNs,
                 dimODEHidden,
                 numODEHidden,
                 delta_t,
                 beta=1.0,
                 numSample=5,
                 K=3,
                 learnstd=False):
        super(NGDCSDE, self).__init__()
        # ODE Part.
        self.ODE = HGDCODE(dimIn,
                           dimRnn,
                           dimState,
                           dimODEHidden,
                           numODEHidden,
                           delta_t,
                           K=K,
                           beta=beta)
        # SDE Part.
        #self.Encoder_post = nn.GRU(dimRnn, dimState, bidirectional=True)                ## [20220113] modified (dimRnn -> dimState)
        #self.Encoder_post = DenseGRUODE(dimRnn, dimState, delta_t)                       ## [20220116] added
        self.Encoder_post = DenseODE(dimRnn, dimState, dimODEHidden, numODEHidden, delta_t) ## [20220117] added
        self.Net_diff = DensNet(dimRnn, dimODEHidden, numODEHidden, dimState)
        self.Net_scale = DensNet(dimRnn, dimODEHidden, numODEHidden, dimState)
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
        _, _, hv_pre, hv_pre_N, hv_post_N, traj_t = self.ODE.forward(
            data_batch, delta_t, return_h=True)
        # compute the inference model.
        batchsize, lenSeq, numNode, dim = hv_pre.size()
        #Z_embed = self.Encoder_post(                                                   ## [20220114] removed
        #    hv_pre.transpose(0, 1).reshape(lenSeq, batchsize * numNode,
        #                                   dim))[0]
        Z_embed = self.Encoder_post(                                                   ## [20220116] added
            hv_pre.transpose(0, 1).reshape(lenSeq, batchsize * numNode,
                                           dim))
        Z_embed = Z_embed.reshape(lenSeq, batchsize, numNode,                          ## [20220114] removed
                                  Z_embed.size(-1)).transpose(0, 1)
        ##Z_embed_pre, _ = torch.chunk(Z_embed, 2, -1)
        ##diffs = 0.1 * self.Net_diff(Z_embed_pre)
        ##scales = torch.tanh(self.Net_scale(Z_embed_pre))
        kld = 0.
        #
        ##Z = self.z0.unsqueeze(0).repeat(numSample, Z_embed.size(0),
        ##                                Z_embed.size(2), 1)
        t, n_frame = data_batch['t'][0], 0                                             ## [20220114] removed
        Zs, Zs_N, Z_priors = [], [], []                                                ## [20220114] removed    
        for n, t_n in enumerate(traj_t):                                               ## [20220114] removed
            #
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
            if n_frame < t.size(0) and t_n > t[n_frame] - 1e-4 * delta_t:                       ## [20220114] removed
        #        Zs_N.append(Z / Z.max(-1, keepdim=True)[0])
                Zs_N.append(Z_embed[:, n, :, :])#.unsqueeze(0).repeat(numSample, 1, 1, 1))        ## [20220113] added
                n_frame += 1                                                                    ## [20220114] removed
        #
        #Zs = torch.stack(Zs, -3)
        Zs_N = torch.stack(Zs_N, -3)                                                            ## [20220114] removed
        #Z_priors = torch.stack(Z_priors, -3)
        #
        #print(hv_pre_N.size(), Zs_N.size())
        #Zsp_N = self.Net_res(hv_pre_N)                                                            ## [20220114] added
        inFeature = torch.cat(
            [hv_pre_N, Zs_N], -1)                     ## [20220114] modified (hv_pre_N.unsqueeze(0).repeat(numSample, 1, 1, 1, 1) -> hv_pre_N)
        #inFeature = Zs_N
        X_pred_residual = self.Output(inFeature)
        X_std = self.Output_std(
            inFeature)[:, 1:, :, :] if self.Output_std is not None else None
        X_pred = X_pred_residual[:, 1:, :, :]                                                
        if return_h:
            return X_pred, hv_pre, hv_pre_N, hv_post_N, Zs, Zs_N, Z_priors, traj_t                
        else:
            return X_pred, X_std, kld, Zs, Zs_N, Z_priors

    def get_loss(self, data_batch):
        X_pred, X_std, kld = self.forward(data_batch,
                                          self.delta_t,
                                          return_h=False)[0:3]
        X = data_batch['values'][:, 1:, :, :]#.unsqueeze(0)
        M_v = data_batch['masks'][:, 1:, :, :]#.unsqueeze(0)
        mse_v = self.criterion(X_pred, X,
                               M_v) if X_std is None else self.criterion(
                                   X_pred, X, M_v, X_std=X_std)
        #print(mse_v.detach().cpu().numpy(), '\t', kld, '\t', self.beta)
        return mse_v, X_pred # + self.beta * kld