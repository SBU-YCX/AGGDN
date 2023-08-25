"""
Author: Yingru Liu
Our models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.vsdn.losses import Gaussian_KL_sigma, Gaussian_NLL
from models.vsdn.ODE_RNN import ODE_RNN_MODEL, ODE_RNN_MODEL_Backward


class VSDN_VAE_FILTER(nn.Module):
    ## VSDN-VAE whose inference model is in filtering mode.
    def __init__(self,
                 input_channels,
                 state_channels,
                 rnn_channels,
                 output_channels,
                 decoder_hidden_channels,
                 prep_hidden_channels,
                 default_deltaT,
                 default_max_T,
                 bias=True,
                 cov_channels=1,
                 cov_hidden_channels=1,
                 dropout_rate=0,
                 Lambda=1.,
                 default_num_samples=5):
        super(VSDN_VAE_FILTER, self).__init__()
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
            nn.Linear(decoder_hidden_channels, output_channels, bias=bias),
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
        self.loss_nll = nn.BCEWithLogitsLoss(reduction='none')
        return

    def forward(self,
                X_enc,
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
        #
        _, loss_gru_ode, path_t, _, path_h, path_hpos = self.gru_ode.forward(
            X_enc, M, delta_t, cov, return_path=True)
        path_h, path_hpos = path_h.unsqueeze(2), path_hpos.unsqueeze(2)
        # Initiate the recurrent operations.
        z = self.covariates_sde_map(cov).unsqueeze(1).repeat(
            1, num_samples, 1, 1)
        current_time, loss_kld, loss_recon = 0., 0., 0.
        path_z, path_p = [], []
        #TODO: REVISE FOR GRAPH NODE.
        for i, Xt in enumerate(X):
            prior_drift = 0.1 * self.drift_model(
                torch.cat([path_h[i].repeat(1,
                                            z.size()[1], 1, 1), z], dim=-1))
            poster_drift = 0.1 * self.drift_model(
                torch.cat([path_hpos[i].repeat(1,
                                               z.size()[1], 1, 1), z],
                          dim=-1))
            diffusion = torch.exp(
                self.diffusion_model(path_h[i].repeat(1,
                                                      z.size()[1], 1, 1)))
            # TODO: deal with explosion.
            if train:
                z = z + delta_t * poster_drift + np.sqrt(
                    delta_t) * diffusion * torch.randn(
                        size=diffusion.size()).cuda()
            else:
                z = z + delta_t * prior_drift + np.sqrt(
                    delta_t) * diffusion * torch.randn(
                        size=diffusion.size()).cuda()
            z_max = z.max(-1, keepdim=True)[0].detach()
            z_min = z.min(-1, keepdim=True)[0].detach()
            z = (z - z_min) / (z_max - z_min)
            p = self.p_model(
                torch.cat([path_h[i].repeat(1,
                                            z.size()[1], 1, 1), z], dim=-1))
            # Compute loss.
            Mt = M[i].mean(-1, keepdim=True)
            errors = (prior_drift - poster_drift) / diffusion
            loss_kld = loss_kld + 0.5 * delta_t * (
                errors**2).mean(1).sum() / path_h.size()[1]
            mean, logvar = p, torch.zeros_like(p).cuda()
            loss_recon_t = Gaussian_NLL(Xt.unsqueeze(1), mean,
                                        logvar)[0].mean(1) * Mt
            loss_recon = loss_recon + loss_recon_t.sum((-1, -2)).mean()
            if return_path:
                path_p.append(p)
                path_z.append(z)
        loss_total = loss_recon + self.Lambda * loss_kld
        if return_path:
            return z, loss_total, loss_recon, loss_kld, np.array(path_t)[0:-1], torch.stack(path_p), path_h, \
                   torch.stack(path_z)
        else:
            return z, loss_total, loss_recon, loss_kld

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
        return loss_total, {
            'NLL': float(loss_total.detach().cpu().numpy()),
            'Recon': float(loss_recon.detach().cpu().numpy()),
            'KLD': float(loss_kld.detach().cpu().numpy())
        }

    def prediction(self, X, M, delta_t, cov):
        _, _, path_t, path_p, path_h, path_z = self.forward(X,
                                                            M,
                                                            delta_t,
                                                            cov,
                                                            return_path=True,
                                                            train=False,
                                                            num_samples=25)
        return path_t, path_p


# TODO:
class VSDN_VAE_SMOOTH(VSDN_VAE_FILTER):
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
                 default_num_samples=5):
        super(VSDN_VAE_SMOOTH,
              self).__init__(input_channels, state_channels, rnn_channels,
                             decoder_hidden_channels, prep_hidden_channels,
                             default_deltaT, default_max_T, bias, cov_channels,
                             cov_hidden_channels, dropout_rate, Lambda,
                             default_num_samples)
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
        current_time, loss_kld, loss_recon = 0., 0., 0.
        path_z, path_p = [], []
        #
        for i, Xt in enumerate(X):
            prior_drift = 0.1 * self.drift_model(
                torch.cat([path_h[i].repeat(1,
                                            z.size()[1], 1), z], dim=-1))
            poster_drift = 0.1 * self.drift_model(
                torch.cat(
                    [(path_h[i] + path_hpos[i]).repeat(1,
                                                       z.size()[1], 1), z],
                    dim=-1))
            diffusion = torch.exp(
                self.diffusion_model(path_h[i].repeat(1,
                                                      z.size()[1], 1)))
            if train:
                z = z + delta_t * poster_drift + np.sqrt(
                    delta_t) * diffusion * torch.randn(
                        size=diffusion.size()).cuda()
            else:
                z = z + delta_t * prior_drift + np.sqrt(
                    delta_t) * diffusion * torch.randn(
                        size=diffusion.size()).cuda()
            p = self.p_model(
                torch.cat([path_h[i].repeat(1,
                                            z.size()[1], 1), z], dim=-1))
            # Compute loss.
            Mt = M[i].mean(-1, keepdim=True)
            errors = (prior_drift - poster_drift) / diffusion
            loss_kld = loss_kld + 0.5 * delta_t * (
                errors**2).mean(1).sum() / path_h.size()[1]
            mean, logvar = torch.chunk(p, 2, dim=-1)
            loss_recon_t = Gaussian_NLL(Xt.unsqueeze(1), mean,
                                        logvar)[0].mean(1) * Mt
            loss_recon = loss_recon + loss_recon_t.sum(-1).mean()
            if return_path:
                path_p.append(p)
                path_z.append(z)
        loss_total = loss_recon + self.Lambda * loss_kld
        if return_path:
            return z, loss_total, np.array(path_t)[0:-1], torch.stack(path_p), path_h, \
                   torch.stack(path_z)
        else:
            return z, loss_total, loss_recon, loss_kld