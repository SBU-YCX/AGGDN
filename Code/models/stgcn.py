"""
Author: Yingru Liu
STGCN.
"""
import torch
import torch.nn as nn
from models.modules import MaskedMSEloss
from models.modules.convolution import GraphConvLayer


class GatedResTempConv(nn.Module):
    "Gated Residual Temporal Convolution Layer"

    def __init__(self, dimIn, dimOut, kernel_size):
        super(GatedResTempConv, self).__init__()
        if dimIn != dimOut:
            self.ConvRes1d = nn.Conv1d(dimIn, dimOut, kernel_size)
        self.ConvGate1d = nn.Conv1d(dimIn, dimOut * 2, kernel_size)
        self.kernel_size, self.dimOut = kernel_size, dimOut
        return

    def forward(self, X):
        batch_size, seqLen, numNode, channel = X.size()
        x = X.permute(0, 2, 3, 1).reshape(-1, channel, seqLen)
        padding_area = torch.zeros(x.size(0), x.size(1),
                                   self.kernel_size - 1).cuda()
        x = torch.cat([padding_area, x], dim=-1)
        feature, gate = torch.chunk(self.ConvGate1d(x)[:, :, :], 2, dim=1)
        feature = feature * torch.sigmoid(gate)
        residual = self.ConvRes1d(x) if hasattr(
            self, 'ConvRes1d') else x[:, :, self.kernel_size - 1:]
        feature = (feature + residual).reshape(batch_size, numNode,
                                               self.dimOut,
                                               seqLen).permute(0, 3, 1, 2)
        return feature


class STConvBlock(nn.Module):
    "STGCN building blocks."

    def __init__(self,
                 dimInNode,
                 dimOutConv1,
                 dimOutConv2,
                 kernel_size,
                 dimGCN,
                 Activation=nn.ReLU):
        super(STConvBlock, self).__init__()
        self.TempConv1 = GatedResTempConv(dimInNode, dimOutConv1, kernel_size)
        self.GCN_Node = GraphConvLayer(dimOutConv1, dimGCN, Activation)
        self.TempConv2 = GatedResTempConv(dimGCN, dimOutConv2, kernel_size)
        return

    def forward(self, X, A_x):
        X1 = self.TempConv1(X)
        Xc = self.GCN_Node(X1, A_x)
        X2 = self.TempConv2(Xc)
        return X2


class STGCN(nn.Module):
    def __init__(self,
                 dimIn,
                 dimOutConv1,
                 dimOutConv2,
                 kernel_size,
                 dimGCN,
                 numBlock,
                 Activation=nn.ReLU,
                 learnstd=False):
        super(STGCN, self).__init__()
        self.net = nn.ModuleList()
        for block_idx in range(numBlock):
            self.net.append(
                STConvBlock(dimIn if not block_idx else dimOutConv2,
                            dimOutConv1, dimOutConv2, kernel_size, dimGCN,
                            Activation))
        self.Output = nn.Linear(dimOutConv2, dimIn)
        self.Output_std = nn.Linear(dimOutConv2, dimIn) if learnstd else None
        self.criterion = MaskedMSEloss()
        return

    def forward(self, data_batch):
        X = data_batch['values'] * data_batch['masks']
        A_x = data_batch['adjacent_matrices']
        for block in self.net:
            X = block(X, A_x)
        X_pred = self.Output(X)
        X_std = self.Output_std(
            X)[:, 0:-1, :, :] if self.Output_std is not None else None
        return X_pred[:, 0:-1, :, :], X_std

    def get_loss(self, data_batch):
        X_pred, X_std = self.forward(data_batch)
        X, M_v = data_batch['values'][:,
                                      1:, :, :], data_batch['masks'][:,
                                                                     1:, :, :]
        mse_v = self.criterion(X_pred, X,
                               M_v) if X_std is None else self.criterion(
                                   X_pred, X, M_v, X_std=X_std)
        return mse_v, X_pred

    def predict(self, data_batch):
        X_pred = self.forward(data_batch)[0]
        return X_pred
