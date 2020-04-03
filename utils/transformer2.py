import torch
import copy
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.modules import activation
from torch import nn
from torch.nn import Dropout, Linear, LayerNorm, ModuleList, Module
from .transformer import positional_encodings
from .gru_helper import generateMask
from .cuda_helper import cuda

class MyMultiheadAttention(Module):

	def __init__(self, d_key, d_value, n_heads, droprate,
				attend_mode,
				window=-1, gumbel_attend=False, use_wo=True, sn=False):

		super().__init__()
		self.module = activation.MultiheadAttention(d_key, n_heads, dropout=droprate, kdim=d_key, vdim=d_value)
		self.attend_mode = attend_mode
		self.window = window
		assert not gumbel_attend
		assert use_wo
		assert not sn

	def forward(self, query, key, value, mask=None, tau=1):
		attn_mask = cuda(torch.ByteTensor(query.shape[0], key.shape[0]), device=query).zero_()
		if self.attend_mode == "only_attend_front":
			assert query.shape[0] == key.shape[0]
			tri = cuda(torch.ones(key.shape[0], key.shape[0]).triu(1), device=query) > 0
			attn_mask = attn_mask or tri
		elif self.attend_mode == "only_attend_back":
			assert query.shape[0] == key.shape[0]
			tri = cuda(torch.ones(key.shape[0], key.shape[0]).tril(1), device=query) > 0
			attn_mask = attn_mask or tri
		elif self.attend_mode == "not_attend_self":
			assert query.shape[0] == key.shape[0]
			eye = cuda(torch.eye(key.shape[1]), device=query) > 0
			attn_mask = attn_mask or eye

		if self.window > 0:
			assert query.shape[1] == key.shape[1]
			window_mask = cuda(torch.ones(key.shape[0], key.shape[0]), device=query)
			window_mask = (window_mask.triu(self.window+1) + window_mask.tril(self.window+1)) > 0
			attn_mask = attn_mask or window_mask

		return self.module.forward(query, key, value, key_padding_mask=mask, attn_mask=attn_mask)

class TransformerEncoder(nn.Module):

	def __init__(self, d_tar, d_hidden, n_heads, n_layers,
				droprate, input_droprate,
				attend_mode="full",
				windows=None, prenorm=False):

		super().__init__()

		if windows is None:
			windows = [-1 for _ in range(n_layers)]

		mods = []
		for i in range(n_layers):
			mods.append(TransformerEncoderLayer(d_tar, d_hidden, n_heads, droprate=droprate, \
				attend_mode=attend_mode,
				window=windows[i], prenorm=prenorm))
		self.layers = nn.ModuleList(mods)

		self.dropout = nn.Dropout(input_droprate)

	def forward(self, x, length=None, mask_tar=None):

		if mask_tar is None and length is not None:
			mask_tar = generateMask(x.shape[1], length, float, device=x).transpose(0, 1)

		x = x.tranpose(0, 1)

		x = self.dropout(x)
		xs = []
		for layer in self.layers:
			x = layer(x, mask_tar=mask_tar)
			xs.append(x.transpose(0, 1))
		return xs

class TransformerDecoder(Module):

	def __init__(self, d_tar, d_src, d_hidden, n_heads, n_layers,
				droprate, input_droprate,
				windows=None, attend_mode="only_attend_front", max_sent_length=-1, prenorm=False):

		super().__init__()

		if windows is None:
			windows = [-1 for _ in range(n_layers)]

		mods = []
		for i in range(n_layers):
			mods.append(TransformerDecoderLayer(d_tar, d_src, d_hidden, n_heads=n_heads, droprate=droprate, \
				window=windows[i], attend_mode=attend_mode, max_sent_length=max_sent_length, prenorm=prenorm))
		self.layers = nn.ModuleList(mods)
		self.attend_mode = attend_mode

		self.dropout = nn.Dropout(input_droprate)

	def forward(self, x, src, src_length=None, tar_length=None, mask_src=None, mask_tar=None):
		# x: batch * seq * feature

		if mask_tar is None and tar_length is not None:
			mask_tar = generateMask(x.shape[1], tar_length, bool, device=x).transpose(0, 1) < 0.5
		if mask_src is None and src_length is not None:
			mask_src = generateMask(src[0].shape[1], src_length, bool, device=x).transpose(0, 1) < 0.5

		x = x.transpose(0, 1)

		x = self.dropout(x)
		xs = []
		for (layer, enc) in zip(self.layers, src):
			x = layer(x, enc.transpose(0, 1), mask_src=mask_src, mask_tar=mask_tar)
			xs.append(x.transpose(0, 1))
		return xs

class TransformerEncoderLayer(Module):

	def __init__(self, d_tar, d_hidden, n_heads, droprate,
				attend_mode, window, prenorm):
		super().__init__()
		self.self_attn = MyMultiheadAttention(d_tar, d_tar, n_heads, droprate=droprate, \
			attend_mode=attend_mode, window=window)
		# Implementation of Feedforward model
		self.linear1 = Linear(d_tar, d_hidden)
		self.dropout = Dropout(droprate)
		self.linear2 = Linear(d_hidden, d_tar)

		self.attend_mode = attend_mode
		self.window = window

		self.norm1 = LayerNorm(d_tar)
		self.norm2 = LayerNorm(d_tar)
		self.dropout1 = Dropout(droprate)
		self.dropout2 = Dropout(droprate)

		self.prenorm = prenorm

		for p in self.parameters():
			if p.dim() > 1:
				xavier_uniform_(p)

	def forward(self, x, mask_tar=None):
		r"""Pass the input through the encoder layer.

		Args:
			src: the sequnce to the encoder layer (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		src2 = self.self_attn(src, src, src, mask=mask_tar)[0]
		if self.prenorm:
			src = src + self.norm1(self.dropout1(src2))
		else:
			src = src + self.dropout1(src2)
			src = self.norm1(src)
		src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
		if self.prenorm:
			src = src + self.norm2(self.dropout2(src2))
		else:
			src = src + self.dropout2(src2)
			src = self.norm2(src)
		return src

class TransformerDecoderLayer(Module):
	def __init__(self, d_tar, d_src, d_hidden, n_heads, droprate, \
				window, attend_mode, max_sent_length, prenorm):

		super().__init__()
		self.self_attn = MyMultiheadAttention(d_tar, d_tar, n_heads, droprate=droprate, attend_mode=attend_mode, window=window)
		self.pos_attn = MyMultiheadAttention(d_tar, d_tar, n_heads, droprate=droprate, attend_mode=attend_mode, window=window)
		self.src_attn = MyMultiheadAttention(d_tar, d_src, n_heads, droprate=droprate, attend_mode="full", window=window)

		# Implementation of Feedforward model
		self.linear1 = Linear(d_tar, d_hidden)
		self.dropout = Dropout(droprate)
		self.linear2 = Linear(d_hidden, d_tar)

		self.norm1 = LayerNorm(d_tar)
		self.norm2 = LayerNorm(d_tar)
		self.norm3 = LayerNorm(d_tar)
		self.norm4 = LayerNorm(d_tar)
		self.dropout1 = Dropout(droprate)
		self.dropout2 = Dropout(droprate)
		self.dropout3 = Dropout(droprate)
		self.dropout4 = Dropout(droprate)

		self.prenorm = prenorm
		self.max_sent_length = max_sent_length

		for p in self.parameters():
			if p.dim() > 1:
				xavier_uniform_(p)

	def forward(self, x, src, mask_src=None, mask_tar=None):
		x2 = self.self_attn(x, x, x, mask_tar)[0]
		if self.prenorm:
			x = x + self.norm1(self.dropout1(x2))
		else:
			x = self.norm1(x + self.dropout1(x2))

		pos_emb = positional_encodings(x.shape[1], x.shape[2], device=x).expand_as(x)
		x2 = self.pos_attn(pos_emb, pos_emb, x, mask_tar)[0]
		if self.prenorm:
			x = x + self.norm2(self.dropout2(x2))
		else:
			x = self.norm2(x + self.dropout2(x2))

		x2 = self.src_attn(x, src, src, mask_src)[0]
		if self.prenorm:
			x = x + self.norm3(self.dropout3(x2))
		else:
			x = self.norm3(x + self.dropout3(x2))

		x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
		if self.prenorm:
			x = x + self.norm4(self.dropout4(x2))
		else:
			x = self.norm4(x + self.dropout4(x2))

		return x
