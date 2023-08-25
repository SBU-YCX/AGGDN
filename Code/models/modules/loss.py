#################################################################################################
#                   LOSS MODULES
#################################################################################################
import torch
import numpy as np
import torch.nn as nn


def Gaussian_NLL(X, mean, logvar, mask=None):
    """
    :param X: shape should be [batch, dim]
    :param mean:
    :param logvar:
    :param mask:
    :return:
    """
    std = torch.exp(0.5 * logvar)
    errors = (X - mean) / std
    constant = np.log(2 * np.pi)
    nll = 0.5 * (errors**2 + logvar +
                 constant) if mask is None else 0.5 * mask * (
                     errors**2 + logvar + constant)
    #nll = 0.5 * (errors**2 + logvar +
    #             constant)
    return nll


class MaskedMSEloss(nn.Module):
    def __init__(self):
        super(MaskedMSEloss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        return

    def forward(self, X_pred, X, mask=None, mean=True, X_std=None):
        if X_std is not None:
            mse = Gaussian_NLL(X, X_pred, X_std, mask)
        else:
            mse = self.criterion(
                X_pred, X) * mask if mask is not None else self.criterion(
                    X_pred, X)
            #mse = self.criterion(
            #    X_pred, X)
        #mae = (torch.abs(X_pred - X) * mask)
        #mape = torch.abs(X_pred - X) / (torch.abs(X) + 1e-6)
        #mape = (mape * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) + 1e-6)
        #
        #print(X.min(), X.max())
        if mean:
            mse = mse.sum((-3, -2, -1)).mean()
            #mae = mae.sum((-3, -2, -1)).mean()
        else:
            mse = mse.sum((-3, -2, -1))
            #mae = mae.sum((-3, -2, -1))
        return mse# + mae) * 0.5
