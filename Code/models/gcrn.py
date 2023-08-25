"""
Author: Yingru Liu
GCGRU , DCGRU.
"""

import torch
import torch.nn as nn
from models.modules import MaskedMSEloss, GCGRUCell, DCGRUCell


# FIXME: Deprecated.
class Discriminator(nn.Module):
    def __init__(self, dimIn, dimRnn, numRNNs, K):
        super(Discriminator, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.Encoder_V = nn.ModuleList()
        self.Output = nn.utils.spectral_norm(nn.Linear(dimRnn, 1))
        for block_idx in range(numRNNs):
            self.Encoder_V.append(
                DCGRUCell(dimIn if not block_idx else dimRnn,
                          dimRnn,
                          K,
                          SN=True))
        return

    def forward(self, X, M_x, A_x, label):
        for block_v in self.Encoder_V:
            X_new = []
            hv = block_v.initiation().repeat(X.size(0), X.size(2), 1)
            #print(X.shape, hv.shape)
            for n in range(X.size(1)):

                hv = block_v(X[:, n, :, :], hv, A_x)
                X_new.append(hv)
            X = torch.stack(X_new, 1)
        label_pred = self.Output(X[:, -1, :, :]).squeeze().mean()
        # loss = self.criterion(label_pred, label)
        return label_pred
