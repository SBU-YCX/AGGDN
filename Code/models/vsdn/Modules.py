"""
Author: Yingru Liu
encoder.
"""
import torch
import math
import numpy as np
import torch.nn as nn


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)


##############################################Revised from official release##############################################
# Copy from https://github.com/edebrouwer/gru_ode_bayes/blob/master/gru_ode_bayes/models.py#L344
# GRU-ODE with input.
class FullGRUODECell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super(FullGRUODECell, self).__init__()
        self.lin_x = nn.Linear(input_size, hidden_size * 3, bias=bias)
        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        return

    def forward(self, x, h):
        """
        Executes one step with GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            x        input values
            h        hidden state (current)
            delta_t  time step

        Returns:
            Updated h
        """
        xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=-1)
        r = torch.sigmoid(xr + self.lin_hr(h))
        z = torch.sigmoid(xz + self.lin_hz(h))
        u = torch.tanh(xh + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh


########################################################################################################################
# Copy from https://github.com/edebrouwer/gru_ode_bayes/blob/master/gru_ode_bayes/models.py#L344
class GRUObservationCell(nn.Module):
    """Implements discrete update based on the received observations."""
    def __init__(self, input_size, hidden_size, prep_hidden, bias=True):
        super(GRUObservationCell, self).__init__()
        self.gru_d = nn.GRUCell(prep_hidden * input_size,
                                hidden_size,
                                bias=bias)
        ## prep layer and its initialization
        std = math.sqrt(2.0 / (2 + prep_hidden))
        self.w_prep = nn.Parameter(
            std * torch.randn(1, 1, input_size, 3, prep_hidden))
        self.bias_prep = nn.Parameter(0.1 +
                                      torch.zeros(input_size, prep_hidden))

        self.input_size = input_size
        self.prep_hidden = prep_hidden
        return

    def forward(self, h, p, X_obs):
        # if losses.sum() != losses.sum():
        #     import ipdb
        #     ipdb.set_trace()
        mean, logvar = torch.chunk(p, 2, dim=-1)
        #print(mean.shape, X_obs.shape)
        gru_input = torch.stack([X_obs, mean, logvar], dim=-1).unsqueeze(-2)
        gru_input = torch.matmul(gru_input,
                                 self.w_prep).squeeze(-2) + self.bias_prep
        gru_input.relu_()
        ## gru_input is (sample x feature x prep_hidden)
        gru_input = (gru_input).contiguous().view(
            -1, self.prep_hidden * self.input_size)
        h_new = self.gru_d(gru_input, h.reshape(-1, h.shape[-1]))
        h_new = h_new.reshape(*h.shape)
        return h_new
