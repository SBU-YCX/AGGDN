"""
Author: Yingru Liu
Our models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.vsdn.Modules import FullGRUODECell, GRUObservationCell, init_weights
from models.vsdn.ODE_RNN import ODE_RNN_MODEL, ODE_RNN_MODEL_Backward
from models.vsdn.losses import Gaussian_KL_sigma, Gaussian_NLL
from torch.autograd import Variable


class VSDN_IWAE_FILTER(nn.Module):
    ## VSDN-VAE whose inference model is in filtering mode.
    def __init__(self,
                 input_channels,
                 state_channels,
                 rnn_channels,
                 decoder_hidden_channels,
                 prep_hidden_channels,
                 default_deltaT,
                 default_max_T,
                 bias=True,
                 cov_channels=1,
                 cov_hidden_channels=1,
                 dropout_rate=0,
                 Lambda=1.,
                 default_num_samples=5,
                 alpha=0.5):
        super(VSDN_IWAE_FILTER, self).__init__()
        self.gru_ode = ODE_RNN_MODEL(input_channels,
                                     rnn_channels,
                                     decoder_hidden_channels,
                                     prep_hidden_channels,
                                     default_deltaT,
                                     default_max_T,
                                     bias=bias,
                                     cov_channels=cov_channels,
                                     cov_hidden_channels=cov_hidden_channels,
                                     mixing=1e-4,
                                     dropout_rate=dropout_rate)
        ### SDE component.
        self.p_model = nn.Sequential(
            nn.Linear(rnn_channels + state_channels,
                      decoder_hidden_channels,
                      bias=bias),
            nn.ReLU(),
            nn.Linear(decoder_hidden_channels, 2 * input_channels, bias=bias),
        )
        self.drift_model = nn.Sequential(
            nn.Linear(rnn_channels + state_channels,
                      decoder_hidden_channels,
                      bias=bias),
            nn.ReLU(),
            nn.Linear(decoder_hidden_channels, state_channels, bias=bias),
            nn.Tanh(),
        )
        self.diffusion_model = nn.Sequential(
            nn.Linear(rnn_channels, decoder_hidden_channels, bias=bias),
            nn.ReLU(),
            nn.Linear(decoder_hidden_channels, state_channels, bias=bias),
        )
        self.covariates_sde_map = nn.Sequential(
            nn.Linear(cov_channels, cov_hidden_channels, bias=bias), nn.ReLU(),
            nn.Linear(cov_hidden_channels, state_channels, bias=bias),
            nn.Tanh())
        self.input_channels = input_channels
        self.default_deltaT, self.default_max_T = default_deltaT, default_max_T
        # hyperparameter of the Neural SDE.
        self.Lambda, self.default_num_samples = Lambda, default_num_samples
        self.alpha = alpha
        return

    def forward(self,
                X,
                M,
                delta_t,
                cov,
                return_path=False,
                num_samples=None,
                train=True):
        """
        Args:
            X          data tensor
            M          mask tensor (1.0 if observed, 0.0 if unobserved)
            delta_t    time step for Euler
            cov        static covariates for learning the first h0
            return_path   whether to return the path of h
            num_samples   num_samples to generate for evidence lower bound.
        Returns:
            h          hidden state at final time (T)
            loss       negative evidence lower bound
            loss_recon reconstruction loss
            loss_kld    kl divergence term of the evidence lower bound.
        """
        num_samples = self.default_num_samples if num_samples is None else num_samples
        _, loss_gru_ode, path_t, _, path_h, path_hpos = self.gru_ode.forward(
            X, M, delta_t, cov, return_path=True)
        path_h, path_hpos = path_h.unsqueeze(2), path_hpos.unsqueeze(2)
        # Initiate the recurrent operations.
        z = self.covariates_sde_map(cov).unsqueeze(1).repeat(1, num_samples, 1)
        current_time, loss_kld, log_wk, log_ll = 0., 0., torch.zeros(path_h.size(1), num_samples).cuda(), \
                                       torch.zeros(path_h.size(1), num_samples).cuda()
        path_z, path_p = [], []
        #
        for i, Xt in enumerate(X):
            prior_drift = 0.1 * self.drift_model(
                torch.cat([path_h[i].repeat(1,
                                            z.size()[1], 1), z], dim=-1))
            poster_drift = 0.1 * self.drift_model(
                torch.cat([path_hpos[i].repeat(1,
                                               z.size()[1], 1), z], dim=-1))
            diffusion = torch.exp(
                self.diffusion_model(path_h[i].repeat(1,
                                                      z.size()[1], 1)))
            # TODO: deal with explosion.
            noise = torch.randn(size=diffusion.size()).cuda()
            if train:
                z = z + delta_t * poster_drift + np.sqrt(
                    delta_t) * diffusion * noise
            else:
                z = z + delta_t * prior_drift + np.sqrt(
                    delta_t) * diffusion * noise
            p = self.p_model(
                torch.cat([
                    path_h[i].repeat(1,
                                     z.size()[1], 1),
                    z / z.max(-1, keepdim=True)[0].detach()
                ],
                          dim=-1))
            # Compute loss.
            Mt = M[i].unsqueeze(1)
            errors = (prior_drift - poster_drift) / diffusion
            dt_log_wk = 0.5 * delta_t * (errors**2)
            dw_log_wk = np.sqrt(delta_t) * errors * noise
            log_wk = log_wk - dt_log_wk.sum(-1) - dw_log_wk.sum(-1)
            loss_kld = loss_kld + dt_log_wk.sum(-1)
            mean, logvar = torch.chunk(p, 2, dim=-1)
            loss_recon_t = Gaussian_NLL(Xt.unsqueeze(1), mean, logvar)[0] * Mt
            log_ll = log_ll - loss_recon_t.sum(-1)
            if return_path:
                path_p.append(p)
                path_z.append(z)
        log_weight = (log_wk + log_ll).detach()
        if train:
            log_weight = log_weight - torch.max(log_weight, -1,
                                                keepdim=True)[0].detach()
            weight = torch.exp(log_weight)
            weight = weight / torch.sum(weight, -1, keepdim=True)
            weight = Variable(weight.data, requires_grad=False)
            loss_iwae = -torch.mean(
                torch.sum(weight * (log_wk + log_ll), dim=-1))
        else:
            loss_iwae = (-torch.logsumexp(log_weight, dim=-1) +
                         np.log(num_samples)).mean()
        loss_vae = (self.Lambda * loss_kld - log_ll).mean()
        loss_total = self.alpha * loss_iwae + (1 - self.alpha) * loss_vae
        if return_path:
            return z, loss_total, np.array(path_t)[0:-1], torch.stack(path_p), path_h, \
                   torch.stack(path_z)
        else:
            return z, loss_total, -log_ll, -log_wk

    def get_loss(self, X, M, cov, deltaT=None, train=True):
        """

        :param times: observation times.
        :param time_ptr: number of points at each observation time.
        :param X: data.
        :param M: mask for observation.
        :param obs_idx: idx of observation path.
        :param cov: covariate for initial state.
        :param deltaT: smallest time interval.
        :return:
        """
        delta_t = self.default_deltaT if deltaT is None else deltaT
        h, loss_total, loss_recon, loss_kld = self.forward(X,
                                                           M,
                                                           delta_t,
                                                           cov,
                                                           train=train)
        return loss_total, {'NLL': float(loss_total.detach().cpu().numpy())}

    def prediction(self, X, M, delta_t, cov):
        _, _, path_t, path_p, path_h, path_z = self.forward(X,
                                                            M,
                                                            delta_t,
                                                            cov,
                                                            return_path=True,
                                                            train=False,
                                                            num_samples=10)
        return path_t, path_p


#
class VSDN_IWAE_SMOOTH(VSDN_IWAE_FILTER):
    ## VSDN-VAE whose inference model is in smoothing mode.
    def __init__(self,
                 input_channels,
                 state_channels,
                 rnn_channels,
                 decoder_hidden_channels,
                 prep_hidden_channels,
                 default_deltaT,
                 default_max_T,
                 bias=True,
                 cov_channels=1,
                 cov_hidden_channels=1,
                 dropout_rate=0,
                 Lambda=1.,
                 default_num_samples=5,
                 alpha=0.5):
        super(VSDN_IWAE_SMOOTH, self).__init__(input_channels,
                                               state_channels,
                                               rnn_channels,
                                               decoder_hidden_channels,
                                               prep_hidden_channels,
                                               default_deltaT,
                                               default_max_T,
                                               bias,
                                               cov_channels,
                                               cov_hidden_channels,
                                               dropout_rate,
                                               Lambda,
                                               default_num_samples,
                                               alpha=alpha)
        self.backward_model = ODE_RNN_MODEL_Backward(
            input_channels,
            rnn_channels,
            decoder_hidden_channels,
            prep_hidden_channels,
            default_deltaT,
            default_max_T,
            bias=bias,
            cov_channels=cov_channels,
            cov_hidden_channels=cov_hidden_channels,
            mixing=1e-4,
            dropout_rate=dropout_rate)
        # self.apply(init_weights)
        return

    def forward(self,
                X,
                M,
                delta_t,
                cov,
                return_path=False,
                num_samples=None,
                train=True):
        """
        Args:
            X          data tensor
            M          mask tensor (1.0 if observed, 0.0 if unobserved)
            delta_t    time step for Euler
            cov        static covariates for learning the first h0
            return_path   whether to return the path of h
            num_samples   num_samples to generate for evidence lower bound.
        Returns:
            h          hidden state at final time (T)
            loss       negative evidence lower bound
            loss_recon reconstruction loss
            loss_kld    kl divergence term of the evidence lower bound.
        """
        num_samples = self.default_num_samples if num_samples is None else num_samples
        _, loss_gru_ode, path_t, _, path_h, path_hpos = self.gru_ode.forward(
            X, M, delta_t, cov, return_path=True)
        path_hpos = self.backward_model.forward(X,
                                                M,
                                                delta_t,
                                                cov,
                                                return_path=True)[-1]
        path_h, path_hpos = path_h.unsqueeze(2), path_hpos.unsqueeze(2)
        # Initiate the recurrent operations.
        z = self.covariates_sde_map(cov).unsqueeze(1).repeat(1, num_samples, 1)
        current_time, loss_kld, log_wk, log_ll = 0., 0., torch.zeros(path_h.size(1), num_samples).cuda(), \
                                                 torch.zeros(path_h.size(1), num_samples).cuda()
        path_z, path_p = [], []
        #
        for i, Xt in enumerate(X):
            prior_drift = 0.1 * self.drift_model(
                torch.cat([path_h[i].repeat(1,
                                            z.size()[1], 1), z], dim=-1))
            poster_drift = 0.1 * self.drift_model(
                torch.cat([path_hpos[i].repeat(1,
                                               z.size()[1], 1), z], dim=-1))
            diffusion = torch.exp(
                self.diffusion_model(path_h[i].repeat(1,
                                                      z.size()[1], 1)))
            #
            noise = torch.randn(size=diffusion.size()).cuda()
            if train:
                z = z + delta_t * poster_drift + np.sqrt(
                    delta_t) * diffusion * noise
            else:
                z = z + delta_t * prior_drift + np.sqrt(
                    delta_t) * diffusion * noise
            p = self.p_model(
                torch.cat([
                    path_h[i].repeat(1,
                                     z.size()[1], 1),
                    z / z.max(-1, keepdim=True)[0].detach()
                ],
                          dim=-1))
            # Compute loss.
            Mt = M[i].unsqueeze(1)
            errors = (prior_drift - poster_drift) / diffusion
            dt_log_wk = 0.5 * delta_t * (errors**2)
            dw_log_wk = np.sqrt(delta_t) * errors * noise
            log_wk = log_wk - dt_log_wk.sum(-1) - dw_log_wk.sum(-1)
            loss_kld = loss_kld + dt_log_wk.sum(-1)
            mean, logvar = torch.chunk(p, 2, dim=-1)
            loss_recon_t = Gaussian_NLL(Xt.unsqueeze(1), mean, logvar)[0] * Mt
            log_ll = log_ll - loss_recon_t.sum(-1)
            if return_path:
                path_p.append(p)
                path_z.append(z)
        log_weight = (log_wk + log_ll).detach()
        if train:
            log_weight = log_weight - torch.max(log_weight, -1,
                                                keepdim=True)[0].detach()
            weight = torch.exp(log_weight)
            weight = weight / torch.sum(weight, -1, keepdim=True)
            weight = Variable(weight.data, requires_grad=False)
            loss_iwae = -torch.mean(
                torch.sum(weight * (log_wk + log_ll), dim=-1))
        else:
            loss_iwae = (-torch.logsumexp(log_weight, dim=-1) +
                         np.log(num_samples)).mean()
        loss_vae = (self.Lambda * loss_kld - log_ll).mean()
        loss_total = self.alpha * loss_iwae + (1 - self.alpha) * loss_vae
        if return_path:
            return z, loss_total, np.array(path_t)[0:-1], torch.stack(path_p), path_h, \
                   torch.stack(path_z)
        else:
            return z, loss_total, -log_ll, -log_wk