"""
	Author 		: Yucheng Xing
	Description : ODE Function
"""


import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm
from models.modules.utils import init_network_weights
from torchdiffeq import odeint, odeint_adjoint


class ODEFunc(nn.Module):
	def __init__(self, ode_func_net):
		super(ODEFunc, self).__init__()
		init_network_weights(ode_func_net)
		self.net = ode_func_net
		self.A = None

	def forward(self, t, h, backwards = False): 
		grad = self.getNet(t, h)
		if backwards: 
			grad = -grad
		return grad

	def getNet(self, t, h):
		return self.net(h, self.A)

	def solve_ode(self, A, ode_method, first_point, time_steps_to_predict, 
				odeint_rtol = 1e-6, odeint_atol = 1e-12, adjoint = False):
		self.A = A
		if adjoint: 
			pred_y = odeint_adjoint(self, first_point, time_steps_to_predict, 
						rtol = odeint_rtol, atol = odeint_atol, 
						method = ode_method)
		else: 
			pred_y = odeint(self, first_point, time_steps_to_predict, 
						rtol = odeint_rtol, atol = odeint_atol, 
						method = ode_method)
		return pred_y

	


'''
class ODEFunc(nn.Module):
	def __init__(self, ode_func_net, A = None):
		super(ODEFunc, self).__init__()
		init_network_weights(ode_func_net)
		self.net = ode_func_net
		self.A = A

	def forward(self, t, h, backwards = False): 
		grad = self.getNet(t, h)
		if backwards: 
			grad = -grad
		return grad

	def getNet(self, t, h):
		return self.net(h, self.A)

	def getPrior(self, h, A, t): 
		return self.getNet(t, h)
'''