"""
Author: Yingru Liu, Yucheng Xing
GRU-ODE.
"""
import torch
import numpy as np
import torch.nn as nn
from models.modules import MaskedMSEloss
from models.modules import DensNet, DDCGRUCell, DDCGRUODECell
from models.modules import DDCNet as DDCODECell


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
                 learnstd=False):
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
        X_pred = self.Output(hv_pre_N * gate)
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
                 learnstd=False):
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
                           beta=beta)
        # SDE Part.
        self.Encoder_pre = nn.GRU(dimRnn, dimRnn, bidirectional = False)
        self.Encoder_post = nn.GRU(dimRnn, dimRnn, bidirectional = True)
        self.Net_diff = DensNet(dimRnn, dimODEHidden, numODEHidden, dimState)
        self.Net_drift = DensNet(dimState, dimODEHidden, numODEHidden, dimState)
        self.Net_prior = DensNet(dimRnn, dimODEHidden,
                                numODEHidden, dimState)
        self.Net_post = DensNet(dimRnn * 2 + dimState, dimODEHidden,
                                numODEHidden, dimState)
        self.Output = nn.Linear(dimRnn + dimState, dimIn)                           
        self.Output_std = nn.Linear(dimRnn + 
                                    dimState, dimIn) if learnstd else None
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
        Z_embed = self.Encoder_post(                                                   
            hv_pre_N.transpose(0, 1).reshape(lenSeq, batchsize * numNode,
                                           dim))[0]
        Z_embed = Z_embed.reshape(lenSeq, batchsize, numNode,                          
                                  Z_embed.size(-1)).transpose(0, 1)
        Z_embed_post = Z_embed
        Z_embed_pre = torch.chunk(Z_embed, 2, -1)[0]
        
        kld = 0.
        #
        Z = self.z0.unsqueeze(0).repeat(numSample, Z_embed_pre.size(0),
                                        Z_embed_pre.size(2), 1)
        t, n_frame = data_batch['t'][0], 0                                             
        Zs_N = []
        for n, t_n in enumerate(traj_t):                                               
            #
            if n_frame < t.size(0) and t_n > t[n_frame] - 1e-4 * delta_t:  
                prior = torch.tanh(self.Net_prior(Z_embed_pre[:, n_frame, :, :]))
                diff = self.Net_diff(Z_embed_pre[:, n_frame, :, :])#0.1 * torch.tanh()
            Z_diff = Z * diff
            Z_prior = 0.5 * prior * (Z_diff ** 2)
            if n_frame < t.size(0) and t_n > t[n_frame] - 1e-4 * delta_t and not return_h: 
                Z_post = self.Net_post(torch.cat([Z_embed_post[:, n_frame, :, :].unsqueeze(0).repeat(
                            numSample, 1, 1, 1), Z], dim=-1))
                Z = Z + Z_post * delta_t + np.sqrt(
                    delta_t) * Z_diff * torch.randn_like(Z_diff)
                kld_t = (Z_post - Z_prior)**2 / torch.max(
                    Z_diff**2,
                    torch.Tensor([1e-2]).cuda())
                kld = kld + delta_t * kld_t.sum((-3, -2, -1)).mean()
            else: 
                Z = Z + Z_prior * delta_t + np.sqrt(
                    delta_t) * Z_diff * torch.randn_like(Z_diff)
            if n_frame < t.size(0) and t_n > t[n_frame] - 1e-4 * delta_t:                      
                Zs_N.append(Z)
                n_frame += 1                                                                    
        #
        Zs_N = torch.stack(Zs_N, -3)                                                            
        inFeature = torch.cat(
            [hv_pre_N.unsqueeze(0).repeat(numSample, 1, 1, 1, 1), Zs_N], -1)                     
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
        X = data_batch['values'][:, 1:, :, :].unsqueeze(0)
        M_v = data_batch['masks'][:, 1:, :, :].unsqueeze(0)
        mse_v = self.criterion(X_pred, X,
                               M_v) if X_std is None else self.criterion(
                                   X_pred, X, M_v, X_std=X_std)
        print(mse_v.detach().cpu().numpy(), '\t', kld.detach().cpu().numpy(), '\t', self.beta)
        return mse_v + self.beta * kld, X_pred.mean(0) # 
