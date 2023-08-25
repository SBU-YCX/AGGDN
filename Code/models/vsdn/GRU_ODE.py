"""
Author (of this code): Yingru Liu
GRU-ODE: https://arxiv.org/abs/1905.12374.
"""
import torch
import torch.nn as nn
import numpy as np
from models.vsdn.Modules import FullGRUODECell, GRUObservationCell, init_weights
from models.vsdn.losses import Gaussian_NLL


class NNFOwithBayesianJumps(torch.nn.Module):
    ## Neural Negative Feedback ODE with Bayesian jumps
    def __init__(self,
                 input_channels,
                 rnn_channels,
                 decoder_hidden_channels,
                 prep_hidden_channels,
                 default_deltaT,
                 default_max_T,
                 bias=True,
                 cov_channels=1,
                 cov_hidden_channels=1,
                 mixing=1.,
                 dropout_rate=0,
                 solver="euler"):
        """
        The smoother variable computes the classification loss as a weighted average of the projection of the latents at each observation.
        impute feeds the parameters of the distribution to GRU-ODE at each step.
        """

        super(NNFOwithBayesianJumps, self).__init__()
        self.p_model = nn.Sequential(
            nn.Linear(rnn_channels, decoder_hidden_channels, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(decoder_hidden_channels, 2 * input_channels, bias=bias),
        )
        self.gru_c = FullGRUODECell(2 * input_channels,
                                    rnn_channels,
                                    bias=bias)
        self.gru_obs = GRUObservationCell(input_channels,
                                          rnn_channels,
                                          prep_hidden_channels,
                                          bias=bias)

        self.covariates_map = nn.Sequential(
            nn.Linear(cov_channels, cov_hidden_channels, bias=bias), nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(cov_hidden_channels, rnn_channels, bias=bias), nn.Tanh())

        assert solver in [
            "euler", "midpoint"
        ], "Solver must be either 'euler' or 'midpoint' or 'dopri5'."

        self.solver = solver
        self.input_channels = input_channels
        self.default_deltaT, self.default_max_T = default_deltaT, default_max_T
        self.mixing = mixing  # mixing hyperparameter for loss_1 and loss_2 aggregation.
        # self.apply(init_weights)
        return

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
        #
        h, loss = self.forward(X, M, delta_t, cov)
        return loss, {'NLL': float(loss.detach().cpu().numpy())}

    def ode_step(self, h, p, delta_t, current_time):
        """Executes a single ODE step."""
        p = torch.zeros_like(p)
        if self.solver == "euler":
            h = h + delta_t * self.gru_c(p, h)
            p = self.p_model(h)

        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(p, h)
            pk = self.p_model(k)

            h = h + delta_t * self.gru_c(pk, k)
            p = self.p_model(h)

        current_time += delta_t
        return h, p, current_time

    def forward(self,
                X,
                M,
                delta_t,
                cov,
                return_path=False,
                smoother=False,
                class_criterion=None,
                labels=None):
        """
        Args:
            X          data tensor
            M          mask tensor (1.0 if observed, 0.0 if unobserved)
            delta_t    time step for Euler
            cov        static covariates for learning the first h0
            return_path   whether to return the path of h
        Returns:
            h          hidden state at final time (T)
            loss       loss of the Gaussian observations
        """
        h = self.covariates_map(cov)
        p = self.p_model(h)
        # Initiate the recurrent operations.
        current_time, loss = 0.0, 0.  # Pre-jump loss and Post-jump loss (KL between p_updated and the actual sample)
        if return_path:
            path_t, path_p, path_h, path_hpos = [], [], [], []

        for i, Xt in enumerate(X):
            ## Propagation of the ODE until next observation
            Mt = (M[i].mean(-1, keepdim=True) > 0).type(
                torch.FloatTensor).cuda()
            h_pre, p_pre, current_time = self.ode_step(h, p, delta_t,
                                                       current_time)
            h = h_pre * (1 - Mt) + self.gru_obs(h, p, Xt) * Mt
            mean, logvar = torch.chunk(p_pre, 2, dim=-1)
            loss_t = (Gaussian_NLL(Xt, mean, logvar)[0] * Mt).sum(-1)
            loss = loss + loss_t
            p = self.p_model(h)
            if return_path:
                path_t.append(current_time)
                path_p.append(p_pre)
                path_h.append(h_pre)
                path_hpos.append(h)
        loss = loss.mean()
        if return_path:
            return h, loss, np.array(path_t), torch.stack(path_p), torch.stack(
                path_h), torch.stack(path_hpos)
        else:
            return h, loss

    def prediction(self, X, M, delta_t, cov):
        _, _, path_t, path_p, path_h, path_hpos = self.forward(
            X, M, delta_t, cov, return_path=True)
        return path_t, path_p
