import torch
import torch.nn.functional as F

from .cuda_helper import cuda

def compile(input_dim, output_dim):
	swap_command = []
	expand_command = []
	input_dim = list(input_dim)

	for i, c in enumerate(output_dim):
		if c == "_":
			expand_command.append(i)
		else:
			oldid = input_dim.index(c)
			swap_command.append(oldid)

	return swap_command, expand_command

RESHAPE_CACHE = {}

def reshape(x, input_dim, output_dim, *args):
	global RESHAPE_CACHE
	assert len(x.shape) == len(input_dim)

	command = RESHAPE_CACHE.get((input_dim, output_dim), None)
	if command is None:
		command = RESHAPE_CACHE[(input_dim, output_dim)] = compile(input_dim, output_dim)

	swap_command, expand_command = command
	x = x.permute(*swap_command)
	expand_size = [-1 for c in output_dim]
	for i, pos in enumerate(expand_command):
		x = x.unsqueeze(pos)
		expand_size[pos] = args[i]

	x = x.expand(*expand_size)
	return x

def cdist2(x, y, eps=1e-12):
    # |x_i - y_j|_2^2 = <x_i - y_j, x_i - y_j> = <x_i, x_i> + <y_j, y_j> - 2*<x_i, y_j>
    x_sq_norm = x.pow(2).sum(dim=-1)
    y_sq_norm = y.pow(2).sum(dim=-1)
    x_dot_y = torch.einsum("ik,jk->ij", x, y)
    sq_dist = x_sq_norm.unsqueeze(dim=1) + y_sq_norm.unsqueeze(dim=0) - 2*x_dot_y
    # For numerical issues
    sq_dist.clamp_(min=eps)
    return torch.sqrt(sq_dist)

def cdist_nobatch(x, y):
	# x = a * b * ... * d_emb
	# y = c * d * ... * d_emb

	d_emb = x.shape[-1]
	x_flatten = x.reshape(-1, d_emb)
	y_flatten = y.reshape(-1, d_emb)

	dis = cdist2(x_flatten, y_flatten)

	return dis.reshape(*(x.shape[:-1] + y.shape[:-1]))
