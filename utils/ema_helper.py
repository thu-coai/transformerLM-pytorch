import torch
from torch import nn

from .module_helper import BaseModule
from .cuda_helper import ones


class EMAHelper(BaseModule):

	def __init__(self, param, parameter_list):
		super().__init__()
		self.args = args = param.args
		self.param = param

		self.parameter_list = list(parameter_list)

		if not parameter_list:
			raise RuntimeError("no ema parameter")

		new_params = [nn.Parameter(torch.Tensor(a.data)) for a in self.parameter_list]
		new_stash = [torch.Tensor(a.data) for a in self.parameter_list]

		self.params = nn.ParameterList(new_params)
		self.stash = new_stash
		self.stash_occupied = False

	def update(self):
		for a, b in zip(self.params, self.parameter_list):
			a.data.mul_(self.args.ema_factor).add_((1 - self.args.ema_factor) * b.data)

	def load(self):
		if self.stash_occupied:
			raise RuntimeError("Last load did not restore")
		self.stash_occupied = True

		for c, b in zip(self.stash, self.parameter_list):
			c.data.copy_(b)

		for a, b in zip(self.params, self.parameter_list):
			b.data.copy_(a)

	def restore(self):
		if not self.stash_occupied:
			raise RuntimeError("Did not load, cannot restore")
		self.stash_occupied = False

		for c, b in zip(self.stash, self.parameter_list):
			b.data.copy_(c)
