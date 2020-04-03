#coding: utf-8
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import numbers

from .cuda_helper import zeros, Tensor, LongTensor
from .gumbel import gumbel_max
from .gru_helper import generateMask
from .storage import Storage


class SequenceBatchNorm(nn.Module):
	def __init__(self, num_features):
		# seqlen * batch * XXXXX * num_features
		super().__init__()

		self.num_features = num_features
		self.bn = nn.BatchNorm1d(num_features)

	def forward(self, incoming, length):
		incoming_shape = incoming.shape
		seqlen = incoming_shape[0]
		batch_num = incoming_shape[1]
		assert self.num_features == incoming_shape[-1]
		assert len(length) == incoming_shape[1]

		incoming = incoming.reshape(seqlen, batch_num, -1, self.num_features)

		arr = []
		for i, l in enumerate(length):
			arr.append(incoming[:l, i])
		alllen = np.sum(length)
		incoming = torch.cat(arr, dim=0)

		incoming = self.bn(incoming.view(-1, self.num_features)).view(alllen, -1, self.num_features)

		#arr = []
		now = 0
		other_dim = incoming.shape[-2]
		res = zeros(seqlen, batch_num, other_dim, self.num_features)
		for i, l in enumerate(length):
			#arr.append(torch.cat([incoming[now:now+l], zeros(seqlen-l, other_dim, self.num_features)], dim=0))
			res[:l, i] = incoming[now:now+l]
			now += l
		#incoming = torch.stack(arr, 1)
		return res.view(*incoming_shape)



class LayerCentralization(nn.Module):
	def __init__(self, normalized_shape, elementwise_affine=True):
		super().__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		self.normalized_shape = tuple(normalized_shape)
		self.elementwise_affine = elementwise_affine
		if self.elementwise_affine:
			self.bias = Parameter(torch.Tensor(*normalized_shape))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		if self.elementwise_affine:
			self.bias.data.zero_()

	def forward(self, input):
		with torch.no_grad():
			reduce_dim = len(input.shape) - len(self.normalized_shape)
			mean = input
			for i in range(reduce_dim):
				mean = mean.mean(-i - 1, keepdim=True)
		return input - mean + self.bias

class BatchCentralization(nn.Module):
	def __init__(self, num_features, momentum=0.1, affine=True,
				 track_running_stats=True):
		super().__init__()
		if isinstance(num_features, numbers.Integral):
			num_features = (num_features,)
		self.num_features = num_features
		self.momentum = momentum
		self.affine = affine
		self.track_running_stats = track_running_stats
		if self.affine:
			self.bias = Parameter(torch.Tensor(*num_features))
		else:
			self.register_parameter('bias', None)
		if self.track_running_stats:
			self.register_buffer('running_mean', torch.zeros(*num_features))
			self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
		else:
			self.register_parameter('running_mean', None)
			self.register_parameter('num_batches_tracked', None)
		self.reset_parameters()

	def reset_running_stats(self):
		if self.track_running_stats:
			self.running_mean.zero_()
			self.num_batches_tracked.zero_()

	def reset_parameters(self):
		self.reset_running_stats()
		if self.affine:
			self.bias.data.zero_()

	def forward(self, input):
		if self.momentum is None:
			exponential_average_factor = 0.0
		else:
			exponential_average_factor = self.momentum

		if self.training and self.track_running_stats:
			if self.num_batches_tracked is not None:
				self.num_batches_tracked += 1
				if self.momentum is None:  # use cumulative moving average
					exponential_average_factor = 1.0 / float(self.num_batches_tracked)
				else:  # use exponential moving average
					exponential_average_factor = self.momentum

		with torch.no_grad():
			if self.training and not self.track_running_stats:
				mean = input
				for _ in range(len(input.shape) - len(self.num_features)):
					mean = mean.mean(dim=0)
				if self.track_running_stats:
					self.running_mean.mul_(1 - exponential_average_factor).add_(exponential_average_factor, mean)
			else:
				mean = self.running_mean

		diff = self.bias - mean
		for _ in range(len(input.shape) - len(self.num_features)):
			diff = diff.unsqueeze(0)
		return input + diff

	def extra_repr(self):
		return '{num_features}, momentum={momentum}, affine={affine}, ' \
			   'track_running_stats={track_running_stats}'.format(**self.__dict__)


class SeqBatchCentralization(nn.Module):
	def __init__(self, num_features, momentum=0.1, affine=True,
				 track_running_stats=True):
		super().__init__()
		if isinstance(num_features, numbers.Integral):
			num_features = (num_features,)
		self.num_features = num_features
		self.momentum = momentum
		self.affine = affine
		self.track_running_stats = track_running_stats
		if self.affine:
			self.bias = Parameter(torch.Tensor(*num_features))
		else:
			self.register_parameter('bias', None)
		if self.track_running_stats:
			self.register_buffer('running_mean', torch.zeros(*num_features))
			self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
		else:
			self.register_parameter('running_mean', None)
			self.register_parameter('num_batches_tracked', None)
		self.reset_parameters()

	def reset_running_stats(self):
		if self.track_running_stats:
			self.running_mean.zero_()
			self.num_batches_tracked.zero_()

	def reset_parameters(self):
		self.reset_running_stats()
		if self.affine:
			self.bias.data.zero_()

	def forward(self, input, sent_length):
		if self.momentum is None:
			exponential_average_factor = 0.0
		else:
			exponential_average_factor = self.momentum

		if self.training and self.track_running_stats:
			if self.num_batches_tracked is not None:
				self.num_batches_tracked += 1
				if self.momentum is None:  # use cumulative moving average
					exponential_average_factor = 1.0 / float(self.num_batches_tracked)
				else:  # use exponential moving average
					exponential_average_factor = self.momentum

		with torch.no_grad():
			if self.training and not self.track_running_stats:
				mask = generateMask(input.shape[1], sent_length).transpose(0, 1)
				for _ in range(self.num_features):
					mask = mask.unsqueeze(-1)
				mean = (input * mask).sum(0).sum(0) / mask.sum()
				if self.track_running_stats:
					self.running_mean.mul_(1 - exponential_average_factor).add_(exponential_average_factor, mean)
			else:
				mean = self.running_mean

		diff = self.bias - mean
		for _ in range(len(input.shape) - len(self.num_features)):
			diff = diff.unsqueeze(0)
		return input + diff

	def extra_repr(self):
		return '{num_features}, momentum={momentum}, affine={affine}, ' \
			   'track_running_stats={track_running_stats}'.format(**self.__dict__)
