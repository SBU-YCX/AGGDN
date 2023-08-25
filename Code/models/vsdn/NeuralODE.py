"""
Author: Yingru Liu
LatentSDE: Scalable gradients for stochastic differential equations.
revised from: https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.vsdn.losses import Gaussian_KL_sigma, Gaussian_NLL
from models.vsdn.ODE_RNN import ODE_RNN_MODEL_Backward
from torch.distributions import Normal, kl_divergence


class LatentODE(nn.Module):
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
                 default_num_samples=5):
        super(LatentODE, self).__init__()
        self.gru_ode = ODE_RNN_MODEL_Backward(
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
        self.p_model = nn.Sequential(
            nn.Linear(state_channels, decoder_hidden_channels, bias=bias),
            nn.ReLU(),
            nn.Linear(decoder_hidden_channels, 2 * input_channels, bias=bias),
        )
        self.sde_init_map = nn.Sequential(
            nn.Linear(rnn_channels, cov_hidden_channels, bias=bias), nn.ReLU(),
            nn.Linear(cov_hidden_channels, 2 * state_channels, bias=bias),
            nn.Tanh())
        self.ode_modal = nn.Sequential(
            nn.Linear(state_channels, decoder_hidden_channels, bias=bias),
            nn.ReLU(),
            nn.Linear(decoder_hidden_channels, state_channels, bias=bias),
            nn.Tanh(),
        )
        self.default_deltaT, self.default_max_T = default_deltaT, default_max_T
        self.default_num_samples = default_num_samples
        self.solver = None
        return

    def forward(self,
                X,
                M,
                delta_t,
                cov,
                return_path=False,
                num_samples=None,
                train=True):
        num_samples = self.default_num_samples if num_samples is None else num_samples
        times = torch.arange(1, X.size()[0] + 1) * delta_t
        h0, _, _, _, path_h, _ = self.gru_ode.forward(X,
                                                      M,
                                                      delta_t,
                                                      cov,
                                                      return_path=True)
        z0_means, z0_logvar = torch.chunk(self.sde_init_map(h0), 2, -1)
        z0_means, z0_logvar = z0_means.unsqueeze(1), z0_logvar.unsqueeze(1)
        eps = torch.randn_like(z0_logvar).cuda().repeat(1, num_samples, 1)
        z = z0_means + torch.exp(0.5 * z0_logvar) * eps
        p = self.p_model(z)
        qy0 = Normal(loc=z0_means, scale=torch.exp(0.5 * z0_logvar))
        py0 = Normal(loc=torch.zeros_like(z0_means).cuda(),
                     scale=torch.ones_like(z0_means).cuda())
        loss_kld = kl_divergence(qy0, py0).sum(1).mean()
        path_z, path_p = [], []
        current_time, loss_recon = 0., 0.
        #
        for i, Xt in enumerate(X):
            z = z + 0.1 * self.ode_modal(z) * delta_t
            p = self.p_model(z)
            # Compute loss.
            Mt = M[i].mean(-1, keepdim=True)
            mean, logvar = torch.chunk(p, 2, dim=-1)
            loss_recon_t = Gaussian_NLL(Xt.unsqueeze(1), mean,
                                        logvar)[0].mean(1) * Mt
            loss_recon = loss_recon + loss_recon_t.sum(-1).mean()
            if return_path:
                path_p.append(p)
                path_z.append(z)
        loss_total = loss_recon + loss_kld
        if return_path:
            return z, loss_total, times, torch.stack(path_p), torch.stack(
                path_z)
        else:
            return z, loss_total, loss_recon, loss_kld

    def get_loss(self, X, M, cov, deltaT=None):
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
        h, loss_total, loss_recon, loss_kld = self.forward(X, M, delta_t, cov)
        return loss_total, {
            'NLL': float(loss_total.detach().cpu().numpy()),
            'Recon': float(loss_recon.detach().cpu().numpy()),
            'KLD': float(loss_kld.detach().cpu().numpy())
        }

    def prediction(self, X, M, delta_t, cov):
        _, _, path_t, path_p, _ = self.forward(X,
                                               M,
                                               delta_t,
                                               cov,
                                               return_path=True,
                                               train=False,
                                               num_samples=10)
        return path_t, path_p