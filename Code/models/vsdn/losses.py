"""
Author: Yingru Liu
loss.
"""
import torch
import torch.nn as nn
import numpy as np


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
    nll = 0.5 * (errors ** 2 + logvar + constant) if mask is None else 0.5 * mask * (errors ** 2 + logvar + constant)
    return nll, errors


# Gaussian KL divergence when network is to output log of variance.
def Gaussian_KL_logvar(mean_1, mean_2, logvar_1, logvar_2, mask=None):
    kl = logvar_2 - logvar_1 + (torch.exp(logvar_1) + (mean_1 - mean_2) ** 2) / (2.0*torch.exp(logvar_2)) - 0.5
    if mask is not None:
        kl = kl * mask
    return kl


# Gaussian KL divergence when network is to output the std.
def Gaussian_KL_sigma(mean_1, mean_2, sigma_1, sigma_2, mask=None):
    kl = (torch.log(sigma_2) - torch.log(sigma_1) + (torch.pow(sigma_1, 2) + torch.pow((mean_1 - mean_2), 2)) / (
            2 * sigma_2 ** 2) - 0.5)
    if mask is not None:
        kl = kl * mask
    return kl