import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils as U
import numpy as np

from utils import zeros, LongTensor, cuda, generateMask,\
			BaseNetwork, DecoderRNN, Storage, gumbel_softmax, flattenSequence, reshape


def positional_encodings(seqlen, feature, length=None, device=None):   # hope to be differentiable

	if length is not None:
		assert feature % 2 == 0
		feature = feature // 2

	assert feature % 2 == 0
	# x: batch x length x feature
	positions = cuda(torch.arange(0, seqlen), device=device).float()
	# length

	# channels
	channels = cuda(torch.arange(0, feature, 2), device=device).float() / feature # 0 2 4 6 ... (256)
	channels = 1 / (10000 ** channels)
	# feature

	# get the positional encoding: batch x target_len
	encodings = positions.unsqueeze(-1) @ channels.unsqueeze(0)  # length x feature
	encodings = torch.cat([torch.sin(encodings), torch.cos(encodings)], -1)
	# length x feature

	if length is None:
		return encodings.unsqueeze(0)
	else:
		encodings_reversed = torch.flip(encodings, [0])
		encodings_back = zeros(len(length), seqlen, feature)
		for i, l in enumerate(length):
			encodings_back[i, :l] = encodings_reversed[-l:]
		return torch.cat([encodings.unsqueeze(0).expand_as(encodings_back), encodings_back], dim=2)

def positional_embeddings(seqlen, first_emb, reverse_emb=None, length=None):
	# first_emb:   max_length * embedding
	encodings = first_emb.unsqueeze(1).expand(-1, seqlen, -1).\
		gather(0, cuda(torch.arange(seqlen)).unsqueeze(-1).expand(-1, first_emb.shape[1]).unsqueeze(0))[0]

	if length is None:
		assert reverse_emb is None
		return encodings.unsqueeze(0)
	else:
		batch_size = len(length)
		reversed_id = np.zeros((batch_size, seqlen))
		for i, l in enumerate(length):
			reversed_id[i, :l] = np.arange(l-1, -1, -1)
		reversed_id = LongTensor(reversed_id)
		encodings_reversed = reverse_emb.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, seqlen, -1).\
								gather(1, reversed_id.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, reverse_emb.shape[-1]))[:, 0]
		return torch.cat([encodings.unsqueeze(0).expand(batch_size, -1, -1), encodings_reversed], dim=2)

class Attention(nn.Module):

	def __init__(self, d_key, d_value, droprate, \
		attend_mode, relative_clip=0, \
		window=-1, gumbel_attend=False):
		super().__init__()
		self.scale = math.sqrt(d_key)
		self.dropout = nn.Dropout(droprate)

		self.relative_clip = relative_clip
		if relative_clip:
			self.relative_size = relative_clip * 2 + 1
			self.key_relative = nn.Embedding(self.relative_size, d_key)
			self.value_relative = nn.Embedding(self.relative_size, d_value)
			self.diag_id = np.zeros(0)
			self.recover_id = np.zeros(0)
			self.d_key = d_key
			self.d_value = d_value

		self.attend_mode = attend_mode
		assert attend_mode in ["full", "only_attend_front", "not_attend_self", "only_attend_back"]
		self.window = window
		self.gumbel_attend = gumbel_attend

	def forward(self, query, key, value, mask=None, tau=1):
		dot_products = (query.unsqueeze(2) * key.unsqueeze(1)).sum(-1)   # batch x query_len x key_len

		if self.relative_clip:
			dot_relative = torch.einsum("ijk,tk->ijt", query, self.key_relative.weight) # batch * query_len * relative_size

			batch_size, query_len, key_len = dot_products.shape

			diag_dim = max(query_len, key_len)
			if self.diag_id.shape[0] < diag_dim:
				self.diag_id = np.zeros((diag_dim, diag_dim))
				for i in range(diag_dim):
					for j in range(diag_dim):
						if i <= j - self.relative_clip:
							self.diag_id[i, j] = 0
						elif i >= j + self.relative_clip:
							self.diag_id[i, j] = self.relative_clip * 2
						else:
							self.diag_id[i, j] = i - j + self.relative_clip
			diag_id = LongTensor(self.diag_id[:query_len, :key_len])

			dot_relative = reshape(dot_relative, "bld", "bl_d", key_len).gather(
					-1, reshape(diag_id, "lm", "_lm_", batch_size, -1)
			)[:, :, :, 0] # batch * query_len * key_len
			dot_products = dot_products + dot_relative

		if self.attend_mode == "only_attend_front":
			assert query.shape[1] == key.shape[1]
			tri = cuda(torch.ones(key.shape[1], key.shape[1]).triu(1), device=query) * 1e9
			dot_products = dot_products - tri.unsqueeze(0)
		elif self.attend_mode == "only_attend_back":
			assert query.shape[1] == key.shape[1]
			tri = cuda(torch.ones(key.shape[1], key.shape[1]).tril(1), device=query) * 1e9
			dot_products = dot_products - tri.unsqueeze(0)
		elif self.attend_mode == "not_attend_self":
			assert query.shape[1] == key.shape[1]
			eye = cuda(torch.eye(key.shape[1]), device=query) * 1e9
			dot_products = dot_products - eye.unsqueeze(0)

		if self.window > 0:
			assert query.shape[1] == key.shape[1]
			window_mask = cuda(torch.ones(key.shape[1], key.shape[1]), device=query)
			window_mask = (window_mask.triu(self.window+1) + window_mask.tril(self.window+1)) * 1e9
			dot_products = dot_products - window_mask.unsqueeze(0)

		if mask is not None:
			dot_products -= (1 - mask) * 1e9

		logits = dot_products / self.scale
		if self.gumbel_attend and self.training:
			probs = gumbel_softmax(logits, tau, dim=-1)
		else:
			probs = torch.softmax(logits, dim=-1)

		probs = probs * ((dot_products <= -5e8).sum(-1, keepdim=True) < dot_products.shape[-1]).float() # batch_size * query_len * key_len
		probs = self.dropout(probs)

		res = torch.matmul(probs, value)  # batch_size * query_len * d_value

		if self.relative_clip:
			if self.recover_id.shape[0] < query_len:
				self.recover_id = np.zeros((query_len, self.relative_size))
				for i in range(query_len):
					for j in range(self.relative_size):
						self.recover_id[i, j] = i + j - self.relative_clip
			recover_id = LongTensor(self.recover_id[:key_len])
			recover_id[recover_id < 0] = key_len
			recover_id[recover_id >= key_len] = key_len

			probs = torch.cat([probs, zeros(batch_size, query_len, 1)], -1)
			relative_probs = probs.gather(
				-1, reshape(recover_id, "qr", "_qr", batch_size)
			) # batch_size * query_len * relative_size
			res = res + torch.einsum("bqr,rd->bqd", relative_probs, self.value_relative.weight) # batch_size * query_len * d_value

		return res

	def init_forward(self):
		i = 0
		def nextStep(query, key, value, mask=None, tau=1):
			nonlocal i

			dot_products = (query.unsqueeze(1) * key).sum(-1)   # batch x query_len x key_len

			if self.attend_mode == "only_attend_front":
				dot_products[:, i+1:] -= 1e9

			if self.window > 0:
				dot_products[:, :max(0, i - self.window)] -= 1e9
				dot_products[:, i + self.window + 1:] -= 1e9

			if self.attend_mode == "not_attend_self":
				dot_products[:, i] -= 1e9

			if mask is not None:
				dot_products -= (1 - mask) * 1e9

			logits = dot_products / self.scale
			if self.gumbel_attend and self.training:
				probs = gumbel_softmax(logits, tau, dim=-1)
			else:
				probs = torch.softmax(logits, dim=-1)

			probs = probs * ((dot_products <= -5e8).sum(-1, keepdim=True) < dot_products.shape[-1]).float()

			i += 1
			return torch.einsum("ij, ijk->ik", self.dropout(probs), value)
		return nextStep

class MultiHeadAttention(nn.Module):

	def __init__(self, d_q, d_k, d_v, d_key, d_value, n_heads, droprate,
				attend_mode, relative_clip=0, \
				window=-1, gumbel_attend=False, use_wo=True, sn=False):
		super().__init__()
		self.attention = Attention(d_key // n_heads, d_value // n_heads, droprate, attend_mode=attend_mode, relative_clip=relative_clip,
			window=window, gumbel_attend=gumbel_attend)
		self.wq = nn.Linear(d_q, d_key, bias=use_wo)
		self.wk = nn.Linear(d_k, d_key, bias=use_wo)
		self.wv = nn.Linear(d_v, d_value, bias=use_wo)
		if sn:
			self.wq = U.spectral_norm(self.wq)
			self.wk = U.spectral_norm(self.wk)
			self.wv = U.spectral_norm(self.wv)
		if use_wo:
			self.wo = nn.Linear(d_value, d_key, bias=use_wo)
			if sn:
				self.wo = U.spectral_norm(self.wo)
		self.use_wo = use_wo
		self.n_heads = n_heads
		self.attend_mode = attend_mode

	def forward(self, query, key, value, mask=None, tau=1):
		query, key, value = self.wq(query), self.wk(key), self.wv(value)   # B x T x D
		B, Tq, D = query.shape
		_, Tk, _ = key.shape
		N = self.n_heads

		query, key, value = (x.reshape(B, -1, N, D//N).transpose(2, 1).reshape(B*N, -1, D//N)
								for x in (query, key, value))
		if mask is not None:
			mask = mask.unsqueeze(1).expand(B, N, Tq, Tk).reshape(B*N, Tq, Tk)
		outputs = self.attention(query, key, value, mask=mask, tau=tau)  # (B x n) x T x (D/n)
		outputs = outputs.reshape(B, N, -1, D//N).transpose(2, 1).reshape(B, -1, D)
		if self.use_wo:
			outputs = self.wo(outputs)
		return outputs

	def init_forward(self, key=None, value=None):

		if key is not None:
			key = self.wk(key)
			value = self.wv(value)
			B, _, D = key.shape
			N = self.n_heads
			keys, values = (x.reshape(B, -1, N, D//N).transpose(2, 1).reshape(B*N, -1, D//N)
									for x in (key, value))
		else:
			assert self.attend_mode == "only_attend_front"
			keys = None
			values = None

		attention_next_step = self.attention.init_forward()

		def nextStep(query, key=None, value=None, mask=None, tau=1, regroup=None):
			nonlocal keys, values, attention_next_step
			N = self.n_heads

			if key is not None:
				key = self.wk(key)
				value = self.wv(value)
				B, D = key.shape

				key, value = (x.reshape(B*N, D//N)
									for x in (key, value))
				if keys is None:
					keys = key.unsqueeze(1)
					values = value.unsqueeze(1)
				else:
					if regroup is not None:
						keys = torch.gather(keys, 0, regroup.unsqueeze(-1).unsqueeze(-1))
						values = torch.gather(keys, 0, regroup.unsqueeze(-1).unsqueeze(-1))
					keys = torch.cat([keys, key.unsqueeze(1)], dim=1)
					values = torch.cat([values, value.unsqueeze(1)], dim=1)

			query = self.wq(query)
			key, value = keys, values
			B, D = query.shape
			_, Tk, _ = key.shape
			query = query.reshape(B*N, D//N)

			if mask is not None:
				mask = mask.unsqueeze(1).expand(B, N, Tk).reshape(B*N, Tk)

			outputs = attention_next_step(query, key, value, mask=mask, tau=tau)
			outputs = outputs.reshape(B, D)
			if self.use_wo:
				outputs = self.wo(outputs)
			return outputs
		return nextStep

class ResidualBlock(nn.Module):

	def __init__(self, layer, d_model, droprate):
		super().__init__()
		self.layer = layer
		self.dropout = nn.Dropout(droprate)
		self.layernorm = nn.LayerNorm(d_model)

	def forward(self, x, *param, **kwargs):
		return self.layernorm(x + self.dropout(self.layer(*param, **kwargs)))

	def init_forward(self, *param, **kwargs):
		layer_next_step = self.layer.init_forward(*param, **kwargs)
		def nextStep(x, *param, **kwargs):
			nonlocal layer_next_step
			assert len(x.shape) == 2
			res = layer_next_step(*param, **kwargs)
			return self.layernorm(x + self.dropout(res))
		return nextStep

class HighwayBlock(nn.Module):

	def __init__(self, layer, d_model, droprate, sn=False):
		super().__init__()
		self.layer = layer
		self.gate = FeedForward(d_model, 1, sn)
		self.dropout = nn.Dropout(droprate)
		self.layernorm = nn.LayerNorm(d_model)

	def forward(self, x, *param, **kwargs):
		g = torch.sigmoid(self.gate(x))
		return self.layernorm(x * g + self.dropout(self.layer(*param, **kwargs)) * (1 - g))

	def init_forward(self, *param, **kwargs):
		layer_next_step = self.layer.init_forward(*param, **kwargs)
		def nextStep(x, *param, **kwargs):
			nonlocal layer_next_step
			g = torch.sigmoid(self.gate(x))
			res = layer_next_step(*param, **kwargs)
			return self.layernorm(x * g + self.dropout(res) * (1 - g))
		return nextStep

class FeedForward(nn.Module):

	def __init__(self, d_model, d_hidden, sn=False):
		super().__init__()
		self.linear1 = nn.Linear(d_model, d_hidden)
		self.linear2 = nn.Linear(d_hidden, d_model)
		if sn:
			self.linear1 = U.spectral_norm(self.linear1)
			self.linear2 = U.spectral_norm(self.linear2)
		self.leaky_relu = nn.LeakyReLU()

	def forward(self, x):
		return self.linear2(self.leaky_relu(self.linear1(x)))

	def init_forward(self):
		def nextStep(x):
			return self.linear2(self.leaky_relu(self.linear1(x)))
		return nextStep

class VocabularyAttention(nn.Module):
	def __init__(self, d_model, sn=False):
		super().__init__()
		self.linear = nn.Linear(d_model, d_model)
		if sn:
			self.linear = U.spectral_norm(self.linear1)

	def forward(self, x, vocab_attention_layer):
		# x: batch_size * seqlen * d_model
		va = vocab_attention_layer(x).softmax(dim=-1) # batch_size * seqlen * vocab_size
		va_weight = torch.einsum("bij,jk->bik", va, vocab_attention_layer.weight) # batch_size * seqlen * d_model
		return self.linear(va_weight)

	def init_forward(self, vocab_attention_layer):
		def nextStep(x):
			# x: seqlen * d_model
			va = self.linear(x).softmax(dim=-1) # seqlen * vocab_size
			va_weight = torch.einsum("ij,jk->ik", va, vocab_attention_layer.weight) # seqlen * d_model
			return self.linear(va_weight)
		return nextStep

class TransformerEncoderLayer(nn.Module):

	def __init__(self, d_tar, d_hidden, n_heads, droprate,
				attend_mode, relative_clip,
				window, use_wo, sn=False, vocabulary_attention=False): # use_wo: use output matrix
		super().__init__()

		self.vocabulary_attention = vocabulary_attention
		if vocabulary_attention:
			self.va_layer = ResidualBlock(
					VocabularyAttention(d_tar, sn=sn),
				d_tar, droprate)

		self.self_attn = ResidualBlock(
			MultiHeadAttention(d_tar, d_tar, d_tar, d_tar, d_tar, n_heads, droprate=droprate,\
				attend_mode=attend_mode, relative_clip=relative_clip,
				window=window, use_wo=use_wo, sn=sn),
			d_tar, droprate)

		self.feedforward = HighwayBlock(FeedForward(d_tar, d_hidden, sn=sn), d_tar, droprate, sn=sn)

	def forward(self, x, mask_tar=None, vocab_attention_layer=None):
		if self.vocabulary_attention:
			x = self.va_layer(x, x, vocab_attention_layer)
		x = self.self_attn(x, x, x, x, mask_tar)
		x = self.feedforward(x, x)
		return x

class TransformerEncoder(nn.Module):

	def __init__(self, d_tar, d_hidden, n_heads, n_layers,
				droprate, input_droprate,
				attend_mode="full", relative_clip=0,
				windows=None, use_wo=True, sn=False, vocabulary_attention=False):

		super().__init__()

		if windows is None:
			windows = [-1 for _ in range(n_layers)]

		mods = []
		for i in range(n_layers):
			mods.append(TransformerEncoderLayer(d_tar, d_hidden, n_heads, droprate=droprate, \
				attend_mode=attend_mode, relative_clip=relative_clip,
				window=windows[i], use_wo=use_wo, sn=sn, vocabulary_attention=vocabulary_attention))
		self.layers = nn.ModuleList(mods)

		self.dropout = nn.Dropout(input_droprate)

	def forward(self, x, length=None, mask_tar=None, vocab_attention_layer=None):

		if mask_tar is None and length is not None:
			mask_tar = generateMask(x.shape[1], length, float, device=x).transpose(0, 1).\
						unsqueeze(1).expand(x.shape[0], x.shape[1], x.shape[1])

		x = self.dropout(x)
		xs = []
		for layer in self.layers:
			x = layer(x, mask_tar=mask_tar, vocab_attention_layer=vocab_attention_layer)
			xs.append(x)
		return xs

class TransformerBiEncoder(nn.Module):

	def __init__(self, d_tar, d_hidden, n_heads, n_layers,
				droprate, input_droprate,
				windows=None, relative_clip=0, use_wo=True, sn=False, vocabulary_attention=False):

		super().__init__()

		self.frontEncoder = TransformerEncoder(d_tar, d_hidden, n_heads, \
			n_layers=n_layers, droprate=droprate, input_droprate=input_droprate,\
			attend_mode="only_attend_front", relative_clip=relative_clip, windows=windows, use_wo=use_wo, sn=sn, \
			vocabulary_attention=vocabulary_attention)
		self.backEncoder = TransformerEncoder(d_tar, d_hidden, n_heads, \
			n_layers=n_layers, droprate=droprate, input_droprate=input_droprate,\
			attend_mode="only_attend_back", relative_clip=relative_clip, windows=windows, use_wo=use_wo, sn=sn, \
			vocabulary_attention=vocabulary_attention)

	def forward(self, x, warning, length=None, mask_tar=None):
		# ones_x = torch.ones_like(x)
		# zeros_x = torch.zeros_like(x)
		# split_pos = 1
		# new_x = torch.cat([ones_x[:, :split_pos], zeros_x[:, split_pos:]], dim=1)

		front_h = self.frontEncoder(x, length, mask_tar)
		back_h = self.backEncoder(x, length, mask_tar)
		assert warning == "I know attend front use self as input, and I will shift the data myself"
		return front_h, back_h

class TransformerDecoderLayer(nn.Module):
	def __init__(self, d_tar, d_src, d_hidden, n_heads, droprate,
				window, attend_mode, gumbel_attend, use_wo, max_sent_length, relative_clip=0, sn=False, vocabulary_attention=False):
		super().__init__()

		if vocabulary_attention:
			self.vocabulary_attention = vocabulary_attention
			self.va_layer = ResidualBlock(
					VocabularyAttention(d_tar, sn=sn),
				d_tar, droprate)

		self.self_attn = ResidualBlock(
			MultiHeadAttention(d_tar, d_tar, d_tar, d_tar, d_tar, n_heads,
					droprate=droprate, attend_mode=attend_mode, window=window, relative_clip=relative_clip,
					use_wo=use_wo, sn=sn),
			d_tar, droprate)
		self.pos_selfattn = ResidualBlock(
			MultiHeadAttention(d_tar, d_tar, d_tar, d_tar, d_tar, n_heads,
					droprate=droprate, attend_mode=attend_mode, window=window, relative_clip=relative_clip,
					gumbel_attend=gumbel_attend, use_wo=use_wo, sn=sn),
			d_tar, droprate)

		self.src_attn = ResidualBlock(
			MultiHeadAttention(d_tar, d_src, d_src, d_tar, d_tar, n_heads,
					droprate=droprate, attend_mode="full", window=window, relative_clip=relative_clip,
					gumbel_attend=gumbel_attend, use_wo=use_wo, sn=sn),  # only noisy gumbel when doing cross-attention
			d_tar, droprate)

		self.d_tar = d_tar
		self.max_sent_length = max_sent_length

		self.feedforward = HighwayBlock(FeedForward(d_tar, d_hidden, sn=sn), d_tar, droprate, sn=sn)

	def forward(self, x, src, mask_src=None, mask_tar=None, vocab_attention_layer=None):
		if self.vocabulary_attention:
			x = self.va_layer(x, x, vocab_attention_layer=vocab_attention_layer)
		x = self.self_attn(x, x, x, x, mask_tar)
		pos_emb = positional_encodings(x.shape[1], x.shape[2], device=x).expand_as(x)
		x = self.pos_selfattn(x, pos_emb, pos_emb, x, mask_tar)
		x = self.src_attn(x, x, src, src, mask_src)
		x = self.feedforward(x, x)
		return x

	def init_forward(self, src, vocab_attention_layer=None):
		if self.vocabulary_attention:
			vocab_attn_step = self.va_layer.init_forward(vocab_attention_layer)
		else:
			vocab_attn_step = None
		self_attn_step = self.self_attn.init_forward()
		pos_selfattn_step = self.pos_selfattn.init_forward()
		src_attn_step = self.src_attn.init_forward(src, src)
		feed_forward_step = self.feedforward.init_forward()
		pos_emb = positional_encodings(self.max_sent_length, self.d_tar).expand(src.shape[0], self.max_sent_length, self.d_tar)
		i = 0

		def nextStep(x, mask_src=None, mask_tar=None, regroup=None, vocab_attention_layer=None):
			nonlocal vocab_attn_step, self_attn_step, pos_selfattn_step, src_attn_step, feed_forward_step, i
			if self.vocabulary_attention:
				x = vocab_attn_step(x)
			x = self_attn_step(x, x, x, x, mask_tar, regroup=regroup)
			x = pos_selfattn_step(x, pos_emb[:, i], pos_emb[:, i], x, mask_tar, regroup=regroup)
			x = src_attn_step(x, x, None, None, mask_src, regroup=regroup)
			x = feed_forward_step(x, x)
			i += 1
			return x

		return nextStep

class TransformerDecoder(DecoderRNN):

	def __init__(self, d_tar, d_src, d_hidden, n_heads, n_layers,
				droprate, input_droprate,
				windows=None, attend_mode="only_attend_front", relative_clip=0,
				gumbel_attend=False, use_wo=True, max_sent_length=-1, sn=False, vocabulary_attention=False):

		super().__init__()

		if windows is None:
			windows = [-1 for _ in range(n_layers)]

		mods = []
		for i in range(n_layers):
			mods.append(TransformerDecoderLayer(d_tar, d_src, d_hidden, n_heads=n_heads, droprate=droprate, \
				window=windows[i], attend_mode=attend_mode, relative_clip=relative_clip, \
					gumbel_attend=gumbel_attend, use_wo=use_wo, \
					max_sent_length=max_sent_length, sn=sn, vocabulary_attention=vocabulary_attention))
		self.layers = nn.ModuleList(mods)
		self.attend_mode = attend_mode

		self.dropout = nn.Dropout(input_droprate)

	def forward(self, x, src, src_length=None, tar_length=None, mask_src=None, mask_tar=None, vocab_attention_layer=None):

		if mask_tar is None and tar_length is not None:
			mask_tar = generateMask(x.shape[1], tar_length, float, device=x).transpose(0, 1).\
						unsqueeze(1).expand(x.shape[0], x.shape[1], x.shape[1])
		if mask_src is None and src_length is not None:
			mask_src = generateMask(src[0].shape[1], src_length, float, device=x).transpose(0, 1).\
						unsqueeze(1).expand(x.shape[0], x.shape[1], src[0].shape[1])

		x = self.dropout(x)
		xs = []
		for (layer, enc) in zip(self.layers, src):
			x = layer(x, enc, mask_src=mask_src, mask_tar=mask_tar, vocab_attention_layer=vocab_attention_layer)
			xs.append(x)
		return xs

	def init_forward(self, src, src_length=None, mask_src=None, vocab_attention_layer=None):

		assert self.attend_mode == "only_attend_front"

		if mask_src is None and src_length is not None:
			mask_src = generateMask(src[0].shape[1], src_length, float, device=src[0]).transpose(0, 1)
		layer_step = []
		for layer, enc in zip(self.layers, src):
			layer_step.append(layer.init_forward(enc, vocab_attention_layer=vocab_attention_layer))

		def nextStep(x, flag=None, regroup=None):
			nonlocal mask_src, layer_step
			xs = []
			for step in layer_step:
				x = step(x, mask_src, regroup=regroup)
				xs.append(x)
			return xs

		return nextStep

	def init_forward_3d(self, src, src_length, mask_src, top_k, vocab_attention_layer=None):

		assert self.attend_mode == "only_attend_front"

		#src batch * seq * hidden

		batch_size = src.shape[0]
		seqlen = src.shape[1]
		if mask_src is None and src_length is not None:
			mask_src = generateMask(src.shape[1], src_length, float, device=src).transpose(0, 1) # batch * seq
		if mask_src is not None:
			mask_src = mask_src.unsqueeze(1).expand(-1, top_k, -1).reshape(batch_size * top_k, -1)
		src = src.unsqueeze(1).expand(-1, top_k, -1, -1).reshape(batch_size * top_k, seqlen, -1)

		step = self.init_forward(src, None, mask_src, vocab_attention_layer)

		def nextStep(x, flag=None, regroup=None):
			nonlocal step, batch_size, top_k
			# regroup: batch * top_k
			regroup = regroup + LongTensor(list(range(batch_size))).unsqueeze(1) * top_k
			regroup = regroup.reshape(-1)
			x = x.reshape(batch_size * top_k, -1)
			x = step(x, regroup=regroup)
			x = x.reshape(batch_size, top_k, -1)

			return x

		return nextStep

	def freerun(self, inp, wLinearLayerCallback, mode='max', input_callback=None, no_unk=True, top_k=10, vocab_attention_layer=None):
		nextStep = self.init_forward(inp.src, inp.get('src_length', None), inp.get('mask_src', None), vocab_attention_layer=vocab_attention_layer)
		return self._freerun(inp, nextStep, wLinearLayerCallback, mode, input_callback, no_unk, top_k)

	def beamsearch(self, inp, top_k, wLinearLayerCallback, input_callback=None, no_unk=True, length_penalty=0.7, vocab_attention_layer=None):
		nextStep = self.init_forward_3d(inp.src, inp.get('src_length', None), inp.get('mask_src', None), top_k, vocab_attention_layer=vocab_attention_layer)
		return self._beamsearch(inp, top_k, nextStep, wLinearLayerCallback, input_callback, no_unk, length_penalty)

class TransformerBiDecoder(nn.Module):

	def __init__(self, d_tar, d_src, d_hidden, n_heads, n_layers,
				droprate, input_droprate,
				windows=None, relative_clip=0, gumbel_attend=False,
				use_wo=True, sn=False):

		super().__init__()

		self.frontDecoder = TransformerDecoder(d_tar, d_src, d_hidden, n_heads=n_heads, \
			n_layers=n_layers, droprate=droprate, input_droprate=input_droprate, windows=windows,\
			attend_mode="only_attend_front", relative_clip=relative_clip, gumbel_attend=gumbel_attend, use_wo=use_wo, sn=sn)
		self.backDecoder = TransformerDecoder(d_tar, d_src, d_hidden, n_heads=n_heads, \
			n_layers=n_layers, droprate=droprate, input_droprate=input_droprate,\
			attend_mode="only_attend_back", relative_clip=relative_clip, gumbel_attend=gumbel_attend, use_wo=use_wo, sn=sn)

	def forward(self, x, src, warning, src_length=None, tar_length=None, mask_src=None, mask_tar=None):
		front_h = self.frontDecoder(x, src, src_length, tar_length, mask_src, mask_tar)
		back_h = self.backDecoder(x, src, src_length, tar_length, mask_src, mask_tar)
		assert warning == "I know attend front use self as input, and I will shift the data myself"
		return front_h, back_h

class TransformerNoSrcDecoderLayer(nn.Module):
	def __init__(self, d_tar, d_hidden, n_heads, droprate,
				window, attend_mode, relative_clip, gumbel_attend, use_wo, max_sent_length, sn=False):
		super().__init__()

		self.self_attn = ResidualBlock(
			MultiHeadAttention(d_tar, d_tar, d_tar, d_tar, d_tar, n_heads,
					droprate=droprate, attend_mode=attend_mode, relative_clip=relative_clip, window=window,
					use_wo=use_wo, sn=sn),
			d_tar, droprate)
		self.pos_selfattn = ResidualBlock(
			MultiHeadAttention(d_tar, d_tar, d_tar, d_tar, d_tar, n_heads,
					droprate=droprate, attend_mode=attend_mode, relative_clip=relative_clip, window=window,
					gumbel_attend=gumbel_attend, use_wo=use_wo, sn=sn),
			d_tar, droprate)
		self.d_tar = d_tar
		self.max_sent_length = max_sent_length
		self.feedforward = HighwayBlock(FeedForward(d_tar, d_hidden, sn=sn), d_tar, droprate, sn=sn)

	def forward(self, x, mask_tar=None):
		x = self.self_attn(x, x, x, x, mask_tar)
		pos_emb = positional_encodings(x.shape[1], x.shape[2], device=x).expand_as(x)
		x = self.pos_selfattn(x, pos_emb, pos_emb, x, mask_tar)
		x = self.feedforward(x, x)
		return x

	def init_forward(self):
		self_attn_step = self.self_attn.init_forward()
		pos_selfattn_step = self.pos_selfattn.init_forward()
		feed_forward_step = self.feedforward.init_forward()
		pos_emb = positional_encodings(self.max_sent_length, self.d_tar)
		i = 0

		def nextStep(x, mask_tar=None, regroup=None):
			nonlocal self_attn_step, pos_selfattn_step, feed_forward_step, i
			x = self_attn_step(x, x, x, x, mask_tar, regroup=regroup)
			pos_emb_expand = pos_emb[:, i].expand(x.shape[0], self.d_tar)
			x = pos_selfattn_step(x, pos_emb_expand, pos_emb_expand, x, mask_tar, regroup=regroup)
			x = feed_forward_step(x, x)
			i += 1
			return x

		return nextStep

class TransformerNoSrcDecoder(DecoderRNN):

	def __init__(self, d_tar, d_hidden, n_heads, n_layers,
				droprate, input_droprate, relative_clip=0,
				windows=None, attend_mode="only_attend_front", gumbel_attend=False, use_wo=True, max_sent_length=-1, sn=False):

		super().__init__()

		if windows is None:
			windows = [-1 for _ in range(n_layers)]

		mods = []
		for i in range(n_layers):
			mods.append(TransformerNoSrcDecoderLayer(d_tar, d_hidden, n_heads=n_heads, droprate=droprate, \
				window=windows[i], relative_clip=relative_clip, attend_mode=attend_mode, gumbel_attend=gumbel_attend, use_wo=use_wo, \
					max_sent_length=max_sent_length, sn=sn))
		self.layers = nn.ModuleList(mods)
		self.attend_mode = attend_mode

		self.dropout = nn.Dropout(input_droprate)

	def forward(self, x, tar_length=None, mask_tar=None):

		if mask_tar is None and tar_length is not None:
			mask_tar = generateMask(x.shape[1], tar_length, float, device=x).transpose(0, 1).\
						unsqueeze(1).expand(x.shape[0], x.shape[1], x.shape[1])

		x = self.dropout(x)
		xs = []
		for layer in self.layers:
			x = layer(x, mask_tar=mask_tar)
			xs.append(x)
		return xs

	def init_forward(self):

		assert self.attend_mode == "only_attend_front"

		layer_step = []
		for layer in self.layers:
			layer_step.append(layer.init_forward())

		def nextStep(x, flag=None, regroup=None):
			nonlocal layer_step
			xs = []
			for step in layer_step:
				x = step(x, regroup=regroup)
				xs.append(x)
			return xs

		return nextStep

	def init_forward_3d(self, top_k):
		assert self.attend_mode == "only_attend_front"
		step = self.init_forward()

		def nextStep(x, flag=None, regroup=None):
			nonlocal step, top_k
			# regroup: batch * top_k
			batch_size = x.shape[0]
			regroup = regroup + LongTensor(list(range(batch_size))).unsqueeze(1) * top_k
			regroup = regroup.reshape(-1)
			x = x.reshape(batch_size * top_k, -1)
			x = step(x, regroup=regroup)
			x = x.reshape(batch_size, top_k, -1)

			return x

		return nextStep

	def freerun(self, inp, wLinearLayerCallback, mode='max', input_callback=None, no_unk=True, top_k=10):
		nextStep = self.init_forward()
		return self._freerun(inp, nextStep, wLinearLayerCallback, mode, input_callback, no_unk, top_k)

	def beamsearch(self, inp, top_k, wLinearLayerCallback, input_callback=None, no_unk=True, length_penalty=0.7):
		nextStep = self.init_forward_3d(top_k)
		return self._beamsearch(inp, top_k, nextStep, wLinearLayerCallback, input_callback, no_unk, length_penalty)
