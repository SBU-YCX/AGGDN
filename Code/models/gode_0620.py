"""
Author: Yingru Liu
GRU-ODE.
"""
import torch
import torchdiffeq
import torchsde
import numpy as np
import torch.nn as nn
from models.modules import MaskedMSEloss
from models.modules import DCGRUCell, DCGRUODECell, GCGRUCell, GCGRUODECell, DensNet, DenseGRUODE
from models.modules import GCNet as GCODECell
from models.modules import DCNet as DCODECell
from models.modules import IdentityNet as IDENTITYLayer
from models.modules import GRUCell
from models.modules import ODEFunc, SDEFunc, DiffeqSolver
from models.modules import linspace_vector


class GCODERNN(nn.Module):
    def __init__(self,
                 dimIn,
                 dimRnn,
                 numRNNs,
                 dimODEHidden,
                 numODEHidden,
                 delta_t,
                 learnstd=False, 
                 ode_method = 'euler'):
        super(GCODERNN, self).__init__()
        self.GRU_V = GCGRUCell(dimIn, dimRnn)
        self.ODE_V = GCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn)
        self.ODE_V_Solver = ODEFunc(self.ODE_V)
        self.ode_method = ode_method
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
        t_iter = range(0, t.size(0))
        t_prev = t[0] - delta_t
        """ODE Part."""
        for n in t_iter:
            # update hv from .
            n_t = ((t[n] + 1e-4 * delta_t - t_prev) / delta_t).int()
            time_points = linspace_vector(t_prev, t[n], n_t).cuda()
            hv1s = self.ODE_V_Solver.solve_ode(A_xt, self.ode_method, hv, time_points)
            #hv_pre.extend(hv1s)
            hv1 = hv1s[-1]
            hv_pre_N.append(hv1)
            hv2 = self.GRU_V(X[:, n, :, :], hv1, A_xt)
            obs_mask = (Mask[:, n, :, :].abs().sum(
                -1, keepdims=True) > 1e-4).type(
                    torch.cuda.FloatTensor)
            hv = hv1 * (1. - obs_mask) + hv2 * obs_mask
            hv_post_N.append(hv)
            # update time.
            t_prev = t[n]
            n += 1
        # stack the features.
        #hv_pre = torch.stack(hv_pre, 1)
        hv_pre_N = torch.stack(hv_pre_N, 1)
        hv_post_N = torch.stack(hv_post_N, 1)
        #
        X_pred = self.Output(hv_pre_N)
        X_std = self.Output_std(
            hv_pre_N)[:, 1:, :, :] if self.Output_std is not None else None
        if return_h:
            return X_pred[:,
                          1:, :, :], X_std, hv_pre_N, hv_post_N
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
                 learnstd=False, 
                 ode_method = 'euler'):
        super(DCODERNN, self).__init__(dimIn,
                                       dimRnn,
                                       numRNNs,
                                       dimODEHidden,
                                       numODEHidden,
                                       delta_t,
                                       learnstd=learnstd, 
                                       ode_method = ode_method)
        self.GRU_V = DCGRUCell(dimIn, dimRnn, K=K)
        self.ODE_V = DCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn, K=K)
        self.ODE_V_Solver = ODEFunc(self.ODE_V)
        self.ode_method = ode_method
        return


class GCGRUODE(GCODERNN):
    def __init__(self, dimIn, dimRnn, numRNNs, delta_t, learnstd=False, ode_method = 'euler'):
        super(GCGRUODE, self).__init__(dimIn,
                                       dimRnn,
                                       numRNNs,
                                       1,
                                       1,
                                       delta_t,
                                       learnstd=learnstd, 
                                       ode_method = ode_method)
        # change the ODE component to GRU cell.
        self.ODE_V = GCGRUODECell(dimRnn)
        self.ODE_V_Solver = ODEFunc(self.ODE_V)
        self.ode_method = ode_method
        return


class DCGRUODE(DCODERNN):
    def __init__(self, dimIn, dimRnn, numRNNs, delta_t, K=3, learnstd=False, ode_method = 'euler'):
        super(DCGRUODE, self).__init__(dimIn,
                                       dimRnn,
                                       numRNNs,
                                       1,
                                       1,
                                       delta_t,
                                       K=K,
                                       learnstd=learnstd, 
                                       ode_method = ode_method)
        self.ODE_V = DCGRUODECell(dimRnn, K=K)
        self.ODE_V_Solver = ODEFunc(self.ODE_V)
        self.ode_method = ode_method
        return


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
        self.GRU_N = GRUCell(dimIn, dimRnn)
        self.ODE_N = DensNet(dimRnn, dimODEHidden, numODEHidden, dimRnn)
        self.ode_method = ode_method
        self.ODE_N_Solver = ODEFunc(self.ODE_N) #DiffeqSolver(ODEFunc(self.ODE_N), self.ode_method)
        self.Output = nn.Linear(dimRnn, dimIn)
        self.Output_std = nn.Linear(dimRnn, dimIn) if learnstd else None
        self.Similarity = nn.TripletMarginLoss()
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
        hv = self.h0.repeat(X.size(0), X.size(2), 1)
        hv_pre, hv_pre_N, hv_post_N = [], [], []
        t_iter = range(0, t.size(0))
        t_prev = t[0] - delta_t
        """ODE Part."""
        for n in t_iter: 
            # update hv, zv
            n_t = ((t[n] + 1e-4 * delta_t - t_prev) // delta_t).int()
            #print(n, t_prev, t[n], n_t, n_t.int())
            time_points = linspace_vector(t_prev, t[n], n_t).cuda()
            hv1s = self.ODE_N_Solver.solve_ode(A_xt, self.ode_method, hv, time_points)#self.ODE_N_Solver(hv, time_points)
            hv1 = hv1s[-1]
            hv_pre_N.append(hv1)
            hv2 = self.GRU_N(X[:, n, :, :], hv1)
            obs_mask = (Mask[:, n, :, :].abs().sum(
                -1, keepdims=True) > 1e-4).type(
                    torch.cuda.FloatTensor)
            hv = hv1 * (1. - obs_mask) + hv2 * obs_mask
            hv_post_N.append(hv)
            t_prev = t[n]
            n += 1
        hv_pre_N = torch.stack(hv_pre_N, 1)
        hv_post_N = torch.stack(hv_post_N, 1)
        #
        X_pred = self.Output(hv_pre_N)
        X_std = self.Output_std(
            hv_pre_N)[:,
                             1:, :, :] if self.Output_std is not None else None
        if return_h:
            return X_pred[:,
                          1:, :, :], X_std, hv_pre_N, hv_post_N
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


class GNODE(nn.Module):
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
        super(GNODE, self).__init__()
        # ODE Part.
        self.GRU_V = DCGRUCell(dimIn, dimRnn, K = K)
        self.ODE_N = DensNet(dimRnn, dimODEHidden, numODEHidden, dimRnn)
        self.ode_method = ode_method
        self.ODE_N_Solver = ODEFunc(self.ODE_N) 
        self.Output = nn.Linear(dimRnn, dimIn)
        self.Output_std = nn.Linear(dimRnn, dimIn) if learnstd else None
        self.Similarity = nn.TripletMarginLoss()
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
        hv = self.h0.repeat(X.size(0), X.size(2), 1)
        hv_pre, hv_pre_N, hv_post_N = [], [], []
        t_iter = range(0, t.size(0))
        t_prev = t[0] - delta_t
        """ODE Part."""
        for n in t_iter: 
            # update hv, zv
            n_t = ((t[n] + 1e-4 * delta_t - t_prev) / delta_t).int()
            time_points = linspace_vector(t_prev, t[n], n_t).cuda()
            hv1s = self.ODE_N_Solver.solve_ode(A_xt, self.ode_method, hv, time_points)
            hv1 = hv1s[-1]
            hv_pre_N.append(hv1)
            hv2 = self.GRU_V(X[:, n, :, :], hv1, A_xt)
            obs_mask = (Mask[:, n, :, :].abs().sum(
                -1, keepdims=True) > 1e-4).type(
                    torch.cuda.FloatTensor)
            hv = hv1 * (1. - obs_mask) + hv2 * obs_mask
            hv_post_N.append(hv)
            t_prev = t[n]
            n += 1
        hv_pre_N = torch.stack(hv_pre_N, 1)
        hv_post_N = torch.stack(hv_post_N, 1)
        #
        X_pred = self.Output(hv_pre_N)
        X_std = self.Output_std(
            hv_pre_N)[:,
                             1:, :, :] if self.Output_std is not None else None
        if return_h:
            return X_pred[:,
                          1:, :, :], X_std, hv_pre_N, hv_post_N
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


class GGRAPHODE(nn.Module):
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
        super(GGRAPHODE, self).__init__()
        # ODE Part.
        self.GRU_V = GCGRUCell(dimIn, dimRnn)
        self.ODE_V = GCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn)
        self.ODE_V_Solver = ODEFunc(self.ODE_V)
        self.ode_method = ode_method
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
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dimRnn)))
        return

    def forward(self, data_batch, delta_t = None, return_h = True):
        if delta_t is None:
            delta_t = self.delta_t
        t = data_batch['t'][0]
        Mask = data_batch['masks']
        X = data_batch['values'] * Mask
        A_xt = data_batch['adjacent_matrices']
        #self.ODE_V_Solver = DiffeqSolver(ODEFunc(self.ODE_V, A_xt), self.ode_method)
        #
        hv = self.h0.repeat(X.size(0), X.size(2), 1)
        hv_pre, hv_pre_N, hv_post_N = [], [], []
        t_iter = range(0, t.size(0))
        t_prev = t[0] - delta_t
        """ODE Part."""
        for n in t_iter: 
            # update hv, zv
            n_t = ((t[n] + 1e-4 * delta_t - t_prev) / delta_t).int()
            #print(n, t_prev, t[n], n_t, n_t.int())
            time_points = linspace_vector(t_prev, t[n], n_t).cuda()
            hv1s = self.ODE_V_Solver.solve_ode(A_xt, self.ode_method, hv, time_points)#self.ODE_V_Solver(hv, time_points)
            hv1 = hv1s[-1]
            hv_pre_N.append(hv1)
            hv2 = self.GRU_V(X[:, n, :, :], hv1, A_xt)
            obs_mask = (Mask[:, n, :, :].abs().sum(
                -1, keepdims=True) > 1e-4).type(
                    torch.cuda.FloatTensor)
            hv = hv1 * (1. - obs_mask) + hv2 * obs_mask
            hv_post_N.append(hv)
            t_prev = t[n]
            n += 1
        hv_pre_N = torch.stack(hv_pre_N, 1)
        hv_post_N = torch.stack(hv_post_N, 1)
        #
        X_pred = self.Output(hv_pre_N)
        X_std = self.Output_std(
            hv_pre_N)[:,
                             1:, :, :] if self.Output_std is not None else None
        if return_h:
            return X_pred[:,
                          1:, :, :], X_std, hv_pre_N, hv_post_N
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


class GRAPHODE(nn.Module):
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
        super(GRAPHODE, self).__init__()
        # ODE Part.
        self.GRU_V = DCGRUCell(dimIn, dimRnn, K=K)
        self.ODE_V = DCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn, K=K)
        self.ODE_V_Solver = ODEFunc(self.ODE_V)
        self.ode_method = ode_method
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
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dimRnn)))
        return

    def forward(self, data_batch, delta_t = None, return_h = True):
        if delta_t is None:
            delta_t = self.delta_t
        t = data_batch['t'][0]
        Mask = data_batch['masks']
        X = data_batch['values'] * Mask
        A_xt = data_batch['adjacent_matrices']
        #self.ODE_V_Solver = DiffeqSolver(ODEFunc(self.ODE_V, A_xt), self.ode_method)
        #
        hv = self.h0.repeat(X.size(0), X.size(2), 1)
        hv_pre, hv_pre_N, hv_post_N = [], [], []
        t_iter = range(0, t.size(0))
        t_prev = t[0] - delta_t
        """ODE Part."""
        for n in t_iter: 
            # update hv, zv
            n_t = ((t[n] + 1e-4 * delta_t - t_prev) / delta_t).int()
            #print(n, t_prev, t[n], n_t, n_t.int())
            time_points = linspace_vector(t_prev, t[n], n_t).cuda()
            hv1s = self.ODE_V_Solver.solve_ode(A_xt, self.ode_method, hv, time_points)#self.ODE_V_Solver(hv, time_points)
            hv1 = hv1s[-1]
            hv_pre_N.append(hv1)
            hv2 = self.GRU_V(X[:, n, :, :], hv1, A_xt)
            obs_mask = (Mask[:, n, :, :].abs().sum(
                -1, keepdims=True) > 1e-4).type(
                    torch.cuda.FloatTensor)
            hv = hv1 * (1. - obs_mask) + hv2 * obs_mask
            hv_post_N.append(hv)
            t_prev = t[n]
            n += 1
        hv_pre_N = torch.stack(hv_pre_N, 1)
        hv_post_N = torch.stack(hv_post_N, 1)
        #
        X_pred = self.Output(hv_pre_N)
        X_std = self.Output_std(
            hv_pre_N)[:,
                             1:, :, :] if self.Output_std is not None else None
        if return_h:
            return X_pred[:,
                          1:, :, :], X_std, hv_pre_N, hv_post_N
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
                 learnstd=False, 
                 ode_method = 'euler', 
                 usegru = True):
        super(HGDCODE, self).__init__()
        # ODE Part.
        self.GRU_V = DCGRUCell(dimIn, dimRnn, K=K)
        self.GRU_G = DCGRUCell(dimIn, dimRnn, K=K)
        self.ODE_V = DCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn, K=K)
        self.ODE_G = DCODECell(dimRnn, dimODEHidden, numODEHidden, dimRnn, K=K)
        self.ODE_V_Solver = ODEFunc(self.ODE_V)
        self.ODE_G_Solver = ODEFunc(self.ODE_G)
        self.ode_method = ode_method
        self.Net_Zout = nn.Sequential(nn.Linear(dimRnn, dimRnn), nn.Tanh(),
                                      nn.Linear(dimRnn, dimRnn), nn.Tanh(),
                                      nn.Linear(dimRnn, dimRnn), nn.Sigmoid())
        self.Output = nn.Linear(dimRnn, dimIn)
        self.Output_std = nn.Linear(dimRnn, dimIn) if learnstd else None
        #self.Output = nn.Sequential(nn.Linear(dimRnn, dimIn), nn.Softplus())
        #self.Output_std = nn.Sequential(nn.Linear(dimRnn, dimIn), nn.Softplus()) if learnstd else None
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

    def forward(self, data_batch, delta_t = None, return_h = True):
        if delta_t is None:
            delta_t = self.delta_t
        t = data_batch['t'][0]
        Mask = data_batch['masks']
        X = data_batch['values'] * Mask
        A_xt = data_batch['adjacent_matrices']
        hv = self.h0.repeat(X.size(0), X.size(2), 1)
        zv = self.z0.repeat(X.size(0), X.size(2), 1)
        hv_pre, hv_pre_N, hv_post_N = [], [], []
        zv_pre, zv_pre_N, zv_post_N = [], [], []
        t_iter = range(0, t.size(0))
        t_prev = t[0] - delta_t
        traj_t = []
        """ODE Part."""
        for n in t_iter: 
            # update hv, zv
            n_t = ((t[n] + 1e-2 * delta_t - t_prev) / (delta_t * 0.99999)).int()
            #print(n, t_prev, t[n], n_t, n_t.int())
            time_points = linspace_vector(t_prev, t[n], n_t).cuda()
            traj_t.extend(time_points + delta_t)
            #print(t[n], t_prev, t[n] + 1e-4 * delta_t - t_prev, n_t, time_points)
            hv1s = self.ODE_V_Solver.solve_ode(A_xt, self.ode_method, hv, time_points)#self.ODE_V_Solver(hv, time_points)
            zv1s = self.ODE_G_Solver.solve_ode(A_xt, self.ode_method, zv, time_points)#self.ODE_V_Solver(hv, time_points)
            hv_pre.extend(hv1s)
            zv_pre.extend(zv1s)
            hv1 = hv1s[-1]
            zv1 = zv1s[-1]
            hv_pre_N.append(hv1)
            zv_pre_N.append(zv1)
            hv2 = self.GRU_V(X[:, n, :, :], hv1, A_xt)
            zv2 = self.GRU_G(X[:, n, :, :], zv1, A_xt)
            obs_mask = (Mask[:, n, :, :].abs().sum(
                -1, keepdims=True) > 1e-4).type(
                    torch.cuda.FloatTensor)
            hv = hv1 * (1. - obs_mask) + hv2 * obs_mask
            zv = zv1 * (1. - obs_mask) + zv2 * obs_mask
            hv_post_N.append(hv)
            zv_post_N.append(zv) 
            t_prev = t[n]
        hv_pre = torch.stack(hv_pre, 1)
        hv_pre_N = torch.stack(hv_pre_N, 1)
        hv_post_N = torch.stack(hv_post_N, 1)
        zv_pre = torch.stack(zv_pre, 1)
        zv_pre_N = torch.stack(zv_pre_N, 1)
        zv_post_N = torch.stack(zv_post_N, 1)
        traj_t = torch.stack(traj_t)
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
                 learnstd=False, 
                 ode_method = 'euler', 
                 sde_method = 'euler_heun', 
                 connect_method = 'r'):
        super(NGDCSDE, self).__init__()
        # ODE Part.
        self.ODE = HGDCODE(dimIn,
                           dimRnn,
                           dimState,
                           dimODEHidden,
                           numODEHidden,
                           delta_t,
                           K=K,
                           beta=beta, 
                           ode_method = ode_method)
        # SDE Part.
        self.connect_method = connect_method
        if self.connect_method == 'r': 
            self.Encoder_sde = nn.GRU(dimRnn, dimRnn, bidirectional = True)
            self.Net_drift = DensNet(dimState, dimODEHidden, numODEHidden, dimState)
            self.Net_diffusion = DensNet(dimState, dimODEHidden, numODEHidden, dimState)
            self.SDE = SDEFunc(self.Net_drift, self.Net_diffusion)
            self.sde_method = sde_method
            self.Net_prior = DensNet(dimRnn + dimState, dimODEHidden, numODEHidden, dimState)
            self.Net_post = DensNet(dimRnn * 2 + dimState, dimODEHidden,
                                    numODEHidden, dimState)
            #self.Net_diff = DensNet(dimRnn, dimODEHidden, numODEHidden, dimState)
            self.Output = nn.Linear(dimRnn + dimState, dimIn)                           
            self.Output_std = nn.Linear(dimRnn + 
                                    dimState, dimIn) if learnstd else None
        else: 
            dimState = dimRnn
            self.Encoder_sde = nn.GRU(dimRnn, dimRnn, bidirectional = True)
            self.Net_drift = DensNet(dimState, dimODEHidden, numODEHidden, dimState)
            self.Net_diffusion = DensNet(dimState, dimODEHidden, numODEHidden, dimState)
            self.SDE = SDEFunc(self.Net_drift, self.Net_diffusion)
            self.sde_method = sde_method
            self.Net_prior = DensNet(dimRnn + dimState, dimODEHidden, numODEHidden, dimState)
            self.Net_post = DensNet(dimRnn * 2 + dimState, dimODEHidden,
                                    numODEHidden, dimState)
            #self.Net_diff = DensNet(dimRnn, dimODEHidden, numODEHidden, dimState)
            self.Output = nn.Linear(dimState, dimIn)                           
            self.Output_std = nn.Linear(dimState, dimIn) if learnstd else None
        self.beta = beta
        self.delta_t = delta_t
        self.criterion = MaskedMSEloss()
        self.numSample = numSample
        # initial state.
        self.z0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dimState)))
        self.dimState = dimState
        return

    def forward(self, data_batch, delta_t=None, numSample=None, return_h=True):
        if delta_t is None:
            delta_t = self.delta_t
        if numSample is None:
            numSample = self.numSample
        #
        ode_pred, ode_std, hv_pre, hv_pre_N, hv_post_N, traj_t = self.ODE.forward(
            data_batch, delta_t, return_h=True)
        # compute the inference model.
        batchsize, lenSeq, numNode, dim = hv_pre.size()
        Z_embed = self.Encoder_sde(                                                   
            hv_pre.transpose(0, 1).reshape(lenSeq, batchsize * numNode,
                                           dim))[0]
        Z_embed = Z_embed.reshape(lenSeq, batchsize, numNode,
                                  Z_embed.size(-1)).transpose(0, 1)
        Z_embed_pre, _ = torch.chunk(Z_embed, 2, -1)
        kld = 0.
        #
        Z = self.z0.unsqueeze(0).repeat(numSample, Z_embed_pre.size(0),
                                        Z_embed_pre.size(2), 1)
        t_iter = range(0, traj_t.size(0))
        t, t_prev, n_frame = data_batch['t'][0], traj_t[0] - delta_t, 0                                          
        Zs, Zs_N = [], []                                                  
        for n in t_iter:                                               
            #
            Z_prior = self.Net_prior(torch.cat([Z_embed_pre[:, n, :, :].unsqueeze(0).repeat(numSample, 1, 1, 1), Z], -1))
            # reach an observation.
            Z_post = self.Net_post(
                torch.cat([
                    Z_embed[:, n, :, :].unsqueeze(0).repeat(
                        numSample, 1, 1, 1), Z
                ], -1))
            n_t = ((traj_t[n] + 1e-4 * delta_t - t_prev) / (delta_t * 0.99999)).int()
            time_points = linspace_vector(t_prev, traj_t[n], 1).cuda()
            #bm = torchsde.BrownianInterval(t0 = t_prev, t1 = traj_t[n], size = (Z.size(0) * Z.size(1) * Z.size(2), self.dimState), device = torch.device('cuda:0'))
            if not return_h:  # training
                Z = self.SDE.solve_sde(self.sde_method, Z_post.reshape(-1, self.dimState), time_points).reshape(Z.size(0), Z.size(1), Z.size(2), Z.size(3))
                kld_t = (Z_post - Z_prior)**2 / torch.max(
                    Z_post**2,
                    torch.Tensor([1e-2]).cuda())
                kld = kld + delta_t * kld_t.sum((-3, -2, -1)).mean()
            else:
                Z = self.SDE.solve_sde(self.sde_method, Z_prior.reshape(-1, self.dimState), time_points).reshape(Z.size(0), Z.size(1), Z.size(2), Z.size(3))
            Zs.append(Z)
            if n_frame < t.size(0) and traj_t[n] > t[n_frame] - 1e-4 * delta_t:                       
                Zs_N.append(Z / Z.max(-1, keepdim=True)[0])
                n_frame += 1    
            t_prev = traj_t[n]                                                                
        #
        Zs = torch.stack(Zs, -3)
        Zs_N = torch.stack(Zs_N, -3)   
        if self.connect_method == 'r':
            inFeature = torch.cat(
            [hv_pre_N.unsqueeze(0).repeat(numSample, 1, 1, 1, 1), Zs_N], -1) 
        else: 
            inFeature = Zs_N                    
        X_pred_residual = self.Output(inFeature)
        X_std = self.Output_std(
            inFeature)[:, :, 1:, :, :] if self.Output_std is not None else None
        X_pred = X_pred_residual[:, :, 1:, :, :]                                                
        if return_h:
            return X_pred.mean(0), hv_pre, hv_pre_N, hv_post_N, Zs, Zs_N, traj_t                
        else:
            return X_pred, X_std, kld, ode_pred, ode_std, Zs, Zs_N 

    def get_loss(self, data_batch):
        X_pred, X_std, kld, ode_pred, ode_std = self.forward(data_batch,
                                          self.delta_t,
                                          return_h=False)[0:5] 
        X = data_batch['values'][:, 1:, :, :].unsqueeze(0)
        M_v = data_batch['masks'][:, 1:, :, :].unsqueeze(0)
        mse_v = self.criterion(X_pred, X,
                               M_v) if X_std is None else self.criterion(
                                   X_pred, X, M_v, X_std=X_std)
        mse_ode = self.criterion(ode_pred, X,
                               M_v) if ode_std is None else self.criterion(
                                   ode_pred, X, M_v, X_std=ode_std)
        #print(mse_v.item(), mse_ode.item(), kld.item())
        return mse_v + mse_ode + self.beta * kld, X_pred
