import torch
import torch.nn as nn


class GraphConvLayer(nn.Module):
    "graph convolutional layer"

    def __init__(self, dimIn, dimOut, activation=nn.ReLU):
        super(GraphConvLayer, self).__init__()
        self.network = nn.Sequential()
        self.network.add_module('linear', nn.Linear(dimIn, dimOut))
        if activation is not None:
            self.network.add_module('activation', activation())
        return

    def forward(self, X, A):
        #X = self.network(X) ##
        tilde_A = A + torch.eye(A.size(-1)).cuda()
        D_x = torch.sqrt(tilde_A.sum(-1))
        L_x = tilde_A / (D_x.unsqueeze(-1) * D_x.unsqueeze(-2) + 1e-12)
        if len(L_x.size()) != len(X.size()):
            L_x = L_x.unsqueeze(1)
        H = self.network(torch.matmul(L_x, X))
        #H = torch.matmul(L_x, X) ##
        return H


class DiffConvLayer(nn.Module):
    "diffusion convolutional layer"

    def __init__(self, dimIn, dimOut, K=3, activation=nn.ReLU, mean=True):
        super(DiffConvLayer, self).__init__()
        self.network = [nn.Linear(dimIn, dimOut) for _ in range(K)]
        for k in range(K):
            setattr(self, 'lin_{}'.format(k), self.network[k])
        self.activation = activation() if activation is not None else None
        self.mean = mean
        self.K = K
        return

    def forward(self, X, A):
        transition = A / (A.sum(-1, keepdims=True) + 1e-12)
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
        #transition = nn.ReLU()(torch.matmul(self.W, transition))
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


# FIXME: Deprecated.
class ADCLayer(nn.Module):
    "graph convolutional layer with attention"

    def __init__(self, dimIn, dimOut, K=3, activation=nn.ReLU):
        super(ADCLayer, self).__init__()
        ## Attention
        self.Linear_V = nn.Linear(dimIn, dimOut, bias=True)
        self.alpha = nn.Sequential(nn.Linear(2 * dimOut, 1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2))
        ## Diffusion
        self.network = [nn.Linear(dimIn, dimOut) for _ in range(K)]
        for k in range(K):
            setattr(self, 'lin_{}'.format(k), self.network[k])
        self.activation = activation() if activation is not None else None
        self.K = K
        return

    def forward(self, X, A):
        #
        N, Xw = X.size(1), self.Linear_V(X)
        att_input = torch.cat([
            Xw.unsqueeze(1).repeat(1, N, 1, 1),
            Xw.unsqueeze(2).repeat(1, 1, N, 1)
        ],
                              dim=-1)
        alpha_matrix = A * torch.exp(self.alpha(att_input).squeeze())
        transition = alpha_matrix / (alpha_matrix.sum(-1, keepdim=True) + 1e-12
                                     )  # normalize.
        #
        Hk = []
        for k in range(self.K):
            Hk.append(self.network[k](X))
            X = torch.matmul(transition, X)
        H = self.activation(torch.stack(
            Hk, -1).sum(-1)) if self.activation is not None else torch.stack(
                Hk, -1).sum(-1)
        return H