import torch
import torch.nn as nn
from models.modules.convolution import DynDiffConvLayer


class DDCGRUODECell(nn.Module):
    "GRU-ODE Cell with diffusion convolution."

    def __init__(self, dimOut, K=3, numNode=33):
        super(DDCGRUODECell, self).__init__(dimOut)
        self.lin_hh = DynDiffConvLayer(dimOut, dimOut, activation=None, K=K, numNode=numNode)
        self.lin_hz = DynDiffConvLayer(dimOut, dimOut, activation=None, K=K, numNode=numNode)
        self.lin_hr = DynDiffConvLayer(dimOut, dimOut, activation=None, K=K, numNode=numNode)
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