import torch
import torch.nn as nn
from models.modules.mlp import SiLU


class DDCGRUCell(nn.Module):
    ""

    def __init__(self, dimIn, dimOut, K=3, SN=False, mean=True, numNode=33):
        super(DDCGRUCell, self).__init__()
        self.K = K
        if SN:
            self.lin_Z = [
                nn.utils.spectral_norm(nn.Linear(dimIn + dimOut, dimOut))
                for _ in range(K)
            ]
            self.lin_R = [
                nn.utils.spectral_norm(nn.Linear(dimIn + dimOut, dimOut))
                for _ in range(K)
            ]
            self.lin_H = [
                nn.utils.spectral_norm(nn.Linear(dimIn + dimOut, dimOut))
                for _ in range(K)
            ]
        else:
            self.lin_Z = [nn.Linear(dimIn + dimOut, dimOut) for _ in range(K)]
            self.lin_R = [nn.Linear(dimIn + dimOut, dimOut) for _ in range(K)]
            self.lin_H = [nn.Linear(dimIn + dimOut, dimOut) for _ in range(K)]
        for k in range(K):
            setattr(self, 'lin_Z_{}'.format(k), self.lin_Z[k])
            setattr(self, 'lin_R_{}'.format(k), self.lin_R[k])
            setattr(self, 'lin_H_{}'.format(k), self.lin_H[k])
        #
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dimOut)))
        self.W = nn.Parameter(0.01 * torch.randn(size=(1, numNode, numNode)))
        self.mean = mean
        return

    def forward(self, x, h, A):
        X = torch.cat([x, h], dim=-1)
        transition = A / (A.sum(-1, keepdims=True) + 1e-12)
        transition = torch.softmax(torch.matmul(self.W, transition), dim=-1)
        Zk, Rk = [], []
        for k in range(self.K):
            Zk.append(self.lin_Z[k](X))
            Rk.append(self.lin_R[k](X))
            X = torch.matmul(transition, X)
        if self.mean:
            Zk = torch.stack(Zk, -1).mean(-1)
            Rk = torch.stack(Rk, -1).mean(-1)
        else:
            Zk = torch.stack(Zk, -1).sum(-1)
            Rk = torch.stack(Rk, -1).sum(-1)
        Z, R = torch.sigmoid(Zk), torch.sigmoid(Rk)
        X = torch.cat([x, R * h], dim = -1)
        Hk = []
        for k in range(self.K):
            Hk.append(self.lin_H[k](X))
            X = torch.matmul(transition, X)
        if self.mean:
            Hk = torch.stack(Hk, -1).mean(-1)
        else:
            Hk = torch.stack(Hk, -1).sum(-1)
        H = torch.tanh(Hk)
        return Z * h + (1 - Z) * H

    def initiation(self, ):
        return self.h0