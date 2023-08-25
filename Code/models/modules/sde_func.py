"""
	Author 		: Yucheng Xing
	Description : SDE Function
"""


import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm
from models.modules.utils import init_network_weights
from torchsde import sdeint


class SDEFunc(nn.Module):
	def __init__(self, drift_net, diffusion_net):#, posterior_net): 
		super(SDEFunc, self).__init__()
		init_network_weights(drift_net)
		self.drift_net = drift_net
		init_network_weights(diffusion_net)
		self.diffusion_net = diffusion_net
		#init_network_weights(posterior_net)
		#self.posterior_net = posterior_net
		self.noise_type = "diagonal"
		self.sde_type = "ito"

	def drift(self, t, h): 
		return self.drift_net(h)

	def diffusion(self, t, h): 
		return self.diffusion_net(h)

	#def posterior(self, t, h): 
	#	return self.posterior_net(h)

	def solve_sde(self, sde_method, first_point, time_steps_to_predict, return_h = True, brownian_motion = None): 
		'''
		if not return_h: 
			pred_y = sdeint(self, 
			            first_point, 
			            time_steps_to_predict, 
			            method = sde_method, 
			            names = {'drift': 'posterior', 'diffusion': 'diffusion'})
		else: 
			'''
		if brownian_motion: 
			pred_y = sdeint(self, 
		            first_point, 
		            time_steps_to_predict, 
		            method = sde_method, 
		            bm = brownian_motion, 
		            names = {'drift': 'drift', 'diffusion': 'diffusion'})#
		else: 
			pred_y = sdeint(self, 
		            first_point, 
		            time_steps_to_predict, 
		            method = sde_method, 
		            names = {'drift': 'drift', 'diffusion': 'diffusion'})#bm = brownian_motion, 
		return pred_y

