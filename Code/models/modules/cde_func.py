"""
	Author 		: Yucheng Xing
	Description : CDE Function
"""


import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm
from models.modules.utils import init_network_weights
from torchcde import cdeint as cdeint


class CDEFunc(nn.Module):
	def __init__(self, ode_func_net):
		super(CDEFunc, self).__init__()
		init_network_weights(cde_func_net)
		self.net = cde_func_net
		self.A = None

	def forward(self, t, h, backwards = False): 
		grad = self.getNet(t, h)
		if backwards: 
			grad = -grad
		return grad

	def getNet(self, t, h):
		return self.net(h, self.A)

	def solve_cde(self, A, cde_method, first_point, time_steps_to_predict, 
				cdeint_rtol = 1e-6, cdeint_atol = 1e-12, adjoint = False):
		self.A = A
		if adjoint: 
			pred_y = cdeint_adjoint(self, first_point, time_steps_to_predict, 
						rtol = odeint_rtol, atol = odeint_atol, 
						method = ode_method)
		else: 
			pred_y = cdeint(self, first_point, time_steps_to_predict, 
						rtol = odeint_rtol, atol = odeint_atol, 
						method = ode_method)
		return pred_y
