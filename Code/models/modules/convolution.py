import torch
import torch.nn as nn


class DynDiffConvLayer(nn.Module):
    "dynamic diffusion convolutional layer"

    def __init__(self, dimIn, dimOut, K=3, activation=nn.ReLU, mean=True, numNode=33):
        super(DynDiffConvLayer, self).__init__()
        self.network = [nn.Linear(dimIn, dimOut) for _ in range(K)]
        for k in range(K):
            setattr(self, 'lin_{}'.format(k), self.network[k])
        self.activation = activation() if activation is not None else None
        self.mean = mean
        self.K = K
        self.W = nn.Parameter(0.01 * torch.randn(size=(1, numNode, numNode)))
        return

    def forward(self, X, A):
        transition = A / (A.sum(-1, keepdims=True) + 1e-12)
        transition = torch.softmax(torch.matmul(self.W, transition), dim=-1)
        Hk = []
        for k in range(self.K):
            Hk.append(self.network[k](X))
            X = torch.matmul(transition, X)
        if self.mean:
            Hk = torch.stack(Hk, -1).mean(-1)
        else:
            Hk = torch.stack(Hk, -1).sum(-1)
        H = self.activation(Hk) if self.activation is not None else Hk
        return H

