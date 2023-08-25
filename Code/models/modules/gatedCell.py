import torch
import torch.nn as nn
from models.modules.mlp import SiLU


#################################################################################################
#                   DENSE MODULES
#################################################################################################
class GRUCell(nn.Module):
    ""

    def __init__(self, dimIn, dimOut, bias=True):
        super(GRUCell, self).__init__()
        self.lin_Z = nn.Sequential(
            nn.Linear(dimIn + dimOut, dimOut, bias=bias), nn.Sigmoid())
        self.lin_R = nn.Sequential(
            nn.Linear(dimIn + dimOut, dimOut, bias=bias), nn.Sigmoid())
        self.lin_H = nn.Sequential(
            nn.Linear(dimIn + dimOut, dimOut, bias=bias), nn.Tanh())
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dimOut)))
        return

    def forward(self, x, h, A=None):
        X = torch.cat([x, h], dim=-1)
        Z, R = self.lin_Z(X), self.lin_R(X)
        H = self.lin_H(torch.cat([x, h * R], dim=-1))
        return Z * h + (1 - Z) * H

    def initiation(self, ):
        return self.h0


class LSTMCell(nn.Module):
    def __init__(self, dimIn, dimOut, bias=True):
        super(LSTMCell, self).__init__()
        self.lin_I = nn.Sequential(
            nn.Linear(dimIn + dimOut, dimOut, bias=bias), nn.Sigmoid())
        self.lin_O = nn.Sequential(
            nn.Linear(dimIn + dimOut, dimOut, bias=bias), nn.Sigmoid())
        self.lin_F = nn.Sequential(
            nn.Linear(dimIn + dimOut, dimOut, bias=bias), nn.Sigmoid())
        self.lin_G = nn.Sequential(
            nn.Linear(dimIn + dimOut, dimOut, bias=bias), nn.Tanh())
        self.act = nn.Tanh()
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dimOut)))
        self.c0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dimOut)))
        return

    def forward(self, x, h, c, A=None):
        X = torch.cat([x, h], dim=-1)
        I, F, G, O = self.lin_I(X), self.lin_F(X), self.lin_G(X), self.lin_O(X)
        C = F * c + I * G
        return O * self.act(C), C

    def initiation(self, ):
        return self.h0, self.c0


#################################################################################################
#                   GRAPH MODULES
#################################################################################################
class GCGRUCell(nn.Module):
    ""

    def __init__(self, dimIn, dimOut, bias=True):
        super(GCGRUCell, self).__init__()
        self.lin_Z = nn.Sequential(
            nn.Linear(dimIn + dimOut, dimOut, bias=bias), nn.Sigmoid())
        self.lin_R = nn.Sequential(
            nn.Linear(dimIn + dimOut, dimOut, bias=bias), nn.Sigmoid())
        self.lin_H = nn.Sequential(
            nn.Linear(dimIn + dimOut, dimOut, bias=bias), nn.Tanh())
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, 1, dimOut)))
        return

    def forward(self, x, h, A):
        X = torch.cat([x, h], dim=-1)
        tilde_A = A + torch.eye(A.size(-1)).cuda()
        D = torch.sqrt(tilde_A.sum(-1))
        L_x = tilde_A / (D.unsqueeze(-1) * D.unsqueeze(-2) + 1e-12)
        preembed = torch.matmul(L_x, X)
        #Z_preembed = self.lin_Z(X) ##
        #R_preembed = self.lin_R(X) ##
        Z, R = self.lin_Z(preembed), self.lin_R(preembed)
        #Z, R = torch.matmul(L_x, Z_preembed), torch.matmul(L_x, R_preembed) ##
        H = self.lin_H(torch.cat([x, h * R], dim=-1))
        return Z * h + (1 - Z) * H

    def initiation(self, ):
        return self.h0


class DCGRUCell(nn.Module):
    ""

    def __init__(self, dimIn, dimOut, K=3, SN=False, mean=True):
        super(DCGRUCell, self).__init__()
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
        self.mean = mean
        return

    def forward(self, x, h, A):
        X = torch.cat([x, h], dim=-1)
        transition = A / (A.sum(-1, keepdims=True) + 1e-12)
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
        #transition = nn.ReLU()(torch.matmul(self.W, transition))
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