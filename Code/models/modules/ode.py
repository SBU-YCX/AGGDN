import torch
import torch.nn as nn
from models.modules.convolution import GraphConvLayer, DiffConvLayer, DynDiffConvLayer


#################################################################################################
#                   Dense MODULES
#################################################################################################
class DenseGRUODECell(nn.Module):
    "GRU-ODE Cell with convolution."

    def __init__(self, dimIn, dimOut):
        super(DenseGRUODECell, self).__init__()
        self.lin_hh = nn.Linear(dimIn + dimOut, dimOut)
        self.lin_hz = nn.Linear(dimIn + dimOut, dimOut)
        self.lin_hr = nn.Linear(dimIn + dimOut, dimOut)
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, dimOut)))
        return

    def forward(self, h, x):
        X = torch.cat([h, x], -1)
        r = torch.sigmoid(self.lin_hr(X))
        z = torch.sigmoid(self.lin_hz(X))
        u = torch.tanh(self.lin_hh(torch.cat([r * h, x], -1)))
        dh = (1 - z) * (u - h)
        return dh

    def initiation(self, ):
        return self.h0


class DenseGRUODE(nn.Module):
    "Whole Sequence GRU-ODE"

    def __init__(self, dimIn, dimOut, delta_t):
        super(DenseGRUODE, self).__init__()
        self.Cell = DenseGRUODECell(dimIn, dimOut)
        self.h0 = nn.Parameter(0.01 * torch.randn(size = (1, dimOut)))
        self.delta_t = delta_t

    def forward(self, X, delta_t = None):
        if not delta_t:
            delta_t = self.delta_t
        T = X.size(0)
        h = self.h0.repeat(X.size(1), 1)
        hs = []
        for t in range(T):
            h = h + self.Cell(h, X[t]) * delta_t
            hs.append(h)
        hs = torch.stack(hs, 1)
        return hs

#################################################################################################
#                   Graph MODULES
#################################################################################################
class GCGRUODECell(nn.Module):
    "GRU-ODE Cell with graph convolution."

    def __init__(self, dimOut):
        super(GCGRUODECell, self).__init__()
        self.lin_hh = GraphConvLayer(dimOut, dimOut, activation=None)
        self.lin_hz = GraphConvLayer(dimOut, dimOut, activation=None)
        self.lin_hr = GraphConvLayer(dimOut, dimOut, activation=None)
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, dimOut)))
        return

    def forward(self, h, A):
        r = torch.sigmoid(self.lin_hr(h, A))
        z = torch.sigmoid(self.lin_hz(h, A))
        u = torch.tanh(self.lin_hh(r * h, A))
        dh = (1 - z) * (u - h)
        return dh

    def initiation(self, ):
        return self.h0


class DCGRUODECell(GCGRUODECell):
    "GRU-ODE Cell with diffusion convolution."

    def __init__(self, dimOut, K=3):
        super(DCGRUODECell, self).__init__(dimOut)
        self.lin_hh = DiffConvLayer(dimOut, dimOut, activation=None, K=K)
        self.lin_hz = DiffConvLayer(dimOut, dimOut, activation=None, K=K)
        self.lin_hr = DiffConvLayer(dimOut, dimOut, activation=None, K=K)
        return


class DDCGRUODECell(GCGRUODECell):
    "GRU-ODE Cell with diffusion convolution."

    def __init__(self, dimOut, K=3, numNode=33):
        super(DDCGRUODECell, self).__init__(dimOut)
        self.lin_hh = DynDiffConvLayer(dimOut, dimOut, activation=None, K=K, numNode=numNode)
        self.lin_hz = DynDiffConvLayer(dimOut, dimOut, activation=None, K=K, numNode=numNode)
        self.lin_hr = DynDiffConvLayer(dimOut, dimOut, activation=None, K=K, numNode=numNode)
        return
