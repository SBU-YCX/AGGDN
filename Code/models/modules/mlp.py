import torch
import torch.nn as nn
from models.modules.convolution import DynDiffConvLayer


class SiLU(nn.Module): 
    @staticmethod
    def forward(self, x): 
        return x * torch.sigmoid(x)


class DensNet(nn.Module):
    def __init__(self,
                 dimIn,
                 dimHidden,
                 numHidden,
                 dimOut,
                 activation=nn.ReLU):
        super(DensNet, self).__init__()
        self.network = nn.Sequential()
        for l in range(numHidden):
            self.network.add_module(
                'layer-{}'.format(l),
                nn.Linear(dimIn if not l else dimHidden, dimHidden))
            self.network.add_module('layer-activation-{}'.format(l),
                                    activation())
        self.network.add_module('output', nn.Linear(dimHidden, dimOut))
        return

    def forward(self, input, A=None):
        return self.network(input)


class DDCNet(nn.Module):
    "An encocer with dynamic diffusion convolutional operation"

    def __init__(self,
                 dimIn,
                 dimHidden,
                 numHidden,
                 dimOut,
                 K=3,
                 activation=nn.ReLU,
                 numNode=33):
        super(DDCNet, self).__init__(dimIn, dimHidden, numHidden, dimOut)
        self.network = nn.ModuleList()
        for l in range(numHidden):
            self.network.append(
                DynDiffConvLayer(dimIn if not l else dimHidden,
                              dimHidden,
                              K,
                              activation=activation, 
                              numNode=numNode))
        self.network.append(nn.Linear(dimHidden, dimOut))
        self.h0 = nn.Parameter(0.01 * torch.randn(size=(1, dimOut)))
        return

    def forward(self, x, A):
        dh = x
        for block in self.network[0:-1]:
            dh = block(dh, A)
        dh = self.network[-1](dh)
        return dh

    def initiation(self, ):
        return self.h0