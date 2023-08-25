"""
	Author 		: Yucheng Xing
	Description : Differential Equation Solver
"""


import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint as odeint, odeint_adjoint


class DiffeqSolver(nn.Module): 
	def __init__(self, ode_func, method, 
				odeint_rtol = 1e-6, odeint_atol = 1e-12, adjoint = True):
				 #odeint_rtol = 1e-4, odeint_atol = 1e-5, adjoint = False):
		super(DiffeqSolver, self).__init__()
		self.ode_method = method
		self.ode_func = ode_func
		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol
		self.adjoint = adjoint

	def forward(self, first_point, time_steps_to_predict, backwards = False): 
		if self.adjoint: 
			pred_y = odeint_adjoint(self.ode_func, first_point, time_steps_to_predict, 
						rtol = self.odeint_rtol, atol = self.odeint_atol, 
						method = self.ode_method)
		else: 
			pred_y = odeint(self.ode_func, first_point, time_steps_to_predict, 
						rtol = self.odeint_rtol, atol = self.odeint_atol, 
						method = self.ode_method)
		return pred_y

	def Prior(self, starting_point_enc, time_steps_to_predict, n_traj_samples = 1): 
		func = self.ode_func.Prior
		pred_y = odeint(func, starting_point_enc, time_steps_to_predict, 
						rtol = self.odeint_rtol, atol = self.odeint_atol, 
						method = self.ode_method)
		return pred_y

