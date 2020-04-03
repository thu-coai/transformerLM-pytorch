import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils as U

from utils import zeros, LongTensor, cuda, generateMask,\
			BaseNetwork, DecoderRNN, Storage, gumbel_softmax, flattenSequence


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

class Attention(nn.Module):

	def __init__(self, d_key, droprate, \
		attend_mode, \
		window=-1, gumbel_attend=False):
		super().__init__()
		self.scale = math.sqrt(d_key)
		self.dropout = nn.Dropout(droprate)
		self.attend_mode = attend_mode
		assert attend_mode in ["full", "only_attend_front", "not_attend_self", "only_attend_back"]
		self.window = window
		self.gumbel_attend = gumbel_attend

	def forward(self, query, key, value, mask=None, tau=1):
		dot_products = (query.unsqueeze(2) * key.unsqueeze(1)).sum(-1)   # batch x query_len x key_len

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

		probs = probs * ((dot_products <= -5e8).sum(-1, keepdim=True) < dot_products.shape[-1]).float()

		return torch.matmul(self.dropout(probs), value)

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

	def __init__(self, d_key, d_value, n_heads, droprate,
				attend_mode,
				window=-1, gumbel_attend=False, use_wo=True, sn=False):
		super().__init__()
		self.attention = Attention(d_key, droprate, attend_mode=attend_mode,
			window=window, gumbel_attend=gumbel_attend)
		self.wq = nn.Linear(d_key, d_key, bias=use_wo)
		self.wk = nn.Linear(d_key, d_key, bias=use_wo)
		self.wv = nn.Linear(d_value, d_value, bias=use_wo)
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

class TransformerEncoderLayer(nn.Module):

	def __init__(self, d_tar, d_hidden, n_heads, droprate,
				attend_mode,
				window, use_wo, sn=False): # use_wo: use output matrix
		super().__init__()

		self.self_attn = ResidualBlock(
			MultiHeadAttention(d_tar, d_tar, n_heads, droprate=droprate,\
				attend_mode=attend_mode,
				window=window, use_wo=use_wo, sn=sn),
			d_tar, droprate)

		self.feedforward = HighwayBlock(FeedForward(d_tar, d_hidden, sn=sn), d_tar, droprate, sn=sn)

	def forward(self, x, mask_tar=None):
		x = self.self_attn(x, x, x, x, mask_tar)
		x = self.feedforward(x, x)
		return x

class TransformerEncoder(nn.Module):

	def __init__(self, d_tar, d_hidden, n_heads, n_layers,
				droprate, input_droprate,
				attend_mode="full",
				windows=None, use_wo=True, sn=False):

		super().__init__()

		if windows is None:
			windows = [-1 for _ in range(n_layers)]

		mods = []
		for i in range(n_layers):
			mods.append(TransformerEncoderLayer(d_tar, d_hidden, n_heads, droprate=droprate, \
				attend_mode=attend_mode,
				window=windows[i], use_wo=use_wo, sn=sn))
		self.layers = nn.ModuleList(mods)

		self.dropout = nn.Dropout(input_droprate)

	def forward(self, x, length=None, mask_tar=None):

		if mask_tar is None and length is not None:
			mask_tar = generateMask(x.shape[1], length, float, device=x).transpose(0, 1).\
						unsqueeze(1).expand(x.shape[0], x.shape[1], x.shape[1])

		x = self.dropout(x)
		xs = []
		for layer in self.layers:
			x = layer(x, mask_tar=mask_tar)
			xs.append(x)
		return xs

class TransformerBiEncoder(nn.Module):

	def __init__(self, d_tar, d_hidden, n_heads, n_layers,
				droprate, input_droprate,
				windows=None, use_wo=True, sn=False):

		super().__init__()

		self.frontEncoder = TransformerEncoder(d_tar, d_hidden, n_heads, \
			n_layers=n_layers, droprate=droprate, input_droprate=input_droprate,\
			attend_mode="only_attend_front", windows=windows, use_wo=use_wo, sn=sn)
		self.backEncoder = TransformerEncoder(d_tar, d_hidden, n_heads, \
			n_layers=n_layers, droprate=droprate, input_droprate=input_droprate,\
			attend_mode="only_attend_back", windows=windows, use_wo=use_wo, sn=sn)


	def forward(self, x, length=None, mask_tar=None):
		front_h = self.frontEncoder(x, length, mask_tar)
		back_h = self.backEncoder(x, length, mask_tar)
		return front_h, back_h

class TransformerDecoderLayer(nn.Module):
	def __init__(self, d_tar, d_src, d_hidden, n_heads, droprate,
				window, attend_mode, gumbel_attend, use_wo, max_sent_length, sn=False):
		super().__init__()

		self.self_attn = ResidualBlock(
			MultiHeadAttention(d_tar, d_tar, n_heads,
					droprate=droprate, attend_mode=attend_mode, window=window,
					use_wo=use_wo, sn=sn),
			d_tar, droprate)
		self.pos_selfattn = ResidualBlock(
			MultiHeadAttention(d_tar, d_tar, n_heads,
					droprate=droprate, attend_mode=attend_mode, window=window,
					gumbel_attend=gumbel_attend, use_wo=use_wo, sn=sn),
			d_tar, droprate)

		self.src_attn = ResidualBlock(
			MultiHeadAttention(d_tar, d_src, n_heads,
					droprate=droprate, attend_mode="full", window=window,
					gumbel_attend=gumbel_attend, use_wo=use_wo, sn=sn),  # only noisy gumbel when doing cross-attention
			d_tar, droprate)
		self.d_tar = d_tar
		self.max_sent_length = max_sent_length

		self.feedforward = HighwayBlock(FeedForward(d_tar, d_hidden, sn=sn), d_tar, droprate, sn=sn)

	def forward(self, x, src, mask_src=None, mask_tar=None):
		x = self.self_attn(x, x, x, x, mask_tar)
		pos_emb = positional_encodings(x.shape[1], x.shape[2], device=x).expand_as(x)
		x = self.pos_selfattn(x, pos_emb, pos_emb, x, mask_tar)
		x = self.src_attn(x, x, src, src, mask_src)
		x = self.feedforward(x, x)
		return x

	def init_forward(self, src):
		self_attn_step = self.self_attn.init_forward()
		pos_selfattn_step = self.pos_selfattn.init_forward()
		src_attn_step = self.src_attn.init_forward(src, src)
		feed_forward_step = self.feedforward.init_forward()
		pos_emb = positional_encodings(self.max_sent_length, self.d_tar).expand(src.shape[0], self.max_sent_length, self.d_tar)
		i = 0

		def nextStep(x, mask_src=None, mask_tar=None, regroup=None):
			nonlocal self_attn_step, pos_selfattn_step, src_attn_step, feed_forward_step, i
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
				windows=None, attend_mode="only_attend_front", gumbel_attend=False, use_wo=True, max_sent_length=-1, sn=False):

		super().__init__()

		if windows is None:
			windows = [-1 for _ in range(n_layers)]

		mods = []
		for i in range(n_layers):
			mods.append(TransformerDecoderLayer(d_tar, d_src, d_hidden, n_heads=n_heads, droprate=droprate, \
				window=windows[i], attend_mode=attend_mode, gumbel_attend=gumbel_attend, use_wo=use_wo, \
					max_sent_length=max_sent_length, sn=sn))
		self.layers = nn.ModuleList(mods)
		self.attend_mode = attend_mode

		self.dropout = nn.Dropout(input_droprate)

	def forward(self, x, src, src_length=None, tar_length=None, mask_src=None, mask_tar=None):

		if mask_tar is None and tar_length is not None:
			mask_tar = generateMask(x.shape[1], tar_length, float, device=x).transpose(0, 1).\
						unsqueeze(1).expand(x.shape[0], x.shape[1], x.shape[1])
		if mask_src is None and src_length is not None:
			mask_src = generateMask(src[0].shape[1], src_length, float, device=x).transpose(0, 1).\
						unsqueeze(1).expand(x.shape[0], x.shape[1], src[0].shape[1])

		x = self.dropout(x)
		xs = []
		for (layer, enc) in zip(self.layers, src):
			x = layer(x, enc, mask_src=mask_src, mask_tar=mask_tar)
			xs.append(x)
		return xs

	def init_forward(self, src, src_length=None, mask_src=None):

		assert self.attend_mode == "only_attend_front"

		if mask_src is None and src_length is not None:
			mask_src = generateMask(src[0].shape[1], src_length, float, device=src[0]).transpose(0, 1)
		layer_step = []
		for layer, enc in zip(self.layers, src):
			layer_step.append(layer.init_forward(enc))

		def nextStep(x, flag=None, regroup=None):
			nonlocal mask_src, layer_step
			xs = []
			for step in layer_step:
				x = step(x, mask_src, regroup=regroup)
				xs.append(x)
			return xs

		return nextStep

	def init_forward_3d(self, src, src_length, mask_src, top_k):

		assert self.attend_mode == "only_attend_front"

		#src batch * seq * hidden

		batch_size = src.shape[0]
		seqlen = src.shape[1]
		if mask_src is None and src_length is not None:
			mask_src = generateMask(src.shape[1], src_length, float, device=src).transpose(0, 1) # batch * seq
		if mask_src is not None:
			mask_src = mask_src.unsqueeze(1).expand(-1, top_k, -1).reshape(batch_size * top_k, -1)
		src = src.unsqueeze(1).expand(-1, top_k, -1, -1).reshape(batch_size * top_k, seqlen, -1)

		step = self.init_forward(src, None, mask_src)

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

	def freerun(self, inp, wLinearLayerCallback, mode='max', input_callback=None, no_unk=True, top_k=10):
		nextStep = self.init_forward(inp.src, inp.get('src_length', None), inp.get('mask_src', None))
		return self._freerun(inp, nextStep, wLinearLayerCallback, mode, input_callback, no_unk, top_k)

	def beamsearch(self, inp, top_k, wLinearLayerCallback, input_callback=None, no_unk=True, length_penalty=0.7):
		nextStep = self.init_forward_3d(inp.src, inp.get('src_length', None), inp.get('mask_src', None), top_k)
		return self._beamsearch(inp, top_k, nextStep, wLinearLayerCallback, input_callback, no_unk, length_penalty)

class TransformerBiDecoder(nn.Module):

	def __init__(self, d_tar, d_src, d_hidden, n_heads, n_layers,
				droprate, input_droprate,
				windows=None, gumbel_attend=False, use_wo=True, sn=False):

		super().__init__()

		self.frontDecoder = TransformerDecoder(d_tar, d_src, d_hidden, n_heads=n_heads, \
			n_layers=n_layers, droprate=droprate, input_droprate=input_droprate, windows=windows,\
			attend_mode="only_attend_front", gumbel_attend=gumbel_attend, use_wo=use_wo, sn=sn)
		self.backDecoder = TransformerDecoder(d_tar, d_src, d_hidden, n_heads=n_heads, \
			n_layers=n_layers, droprate=droprate, input_droprate=input_droprate,\
			attend_mode="only_attend_back", gumbel_attend=gumbel_attend, use_wo=use_wo, sn=sn)

	def forward(self, x, src, src_length=None, tar_length=None, mask_src=None, mask_tar=None):
		front_h = self.frontDecoder(x, src, src_length, tar_length, mask_src, mask_tar)
		back_h = self.backDecoder(x, src, src_length, tar_length, mask_src, mask_tar)
		return front_h, back_h

class TransformerNoSrcDecoderLayer(nn.Module):
	def __init__(self, d_tar, d_hidden, n_heads, droprate,
				window, attend_mode, gumbel_attend, use_wo, max_sent_length, sn=False):
		super().__init__()

		self.self_attn = ResidualBlock(
			MultiHeadAttention(d_tar, d_tar, n_heads,
					droprate=droprate, attend_mode=attend_mode, window=window,
					use_wo=use_wo, sn=sn),
			d_tar, droprate)
		self.pos_selfattn = ResidualBlock(
			MultiHeadAttention(d_tar, d_tar, n_heads,
					droprate=droprate, attend_mode=attend_mode, window=window,
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
				droprate, input_droprate,
				windows=None, attend_mode="only_attend_front", gumbel_attend=False, use_wo=True, max_sent_length=-1, sn=False):

		super().__init__()

		if windows is None:
			windows = [-1 for _ in range(n_layers)]

		mods = []
		for i in range(n_layers):
			mods.append(TransformerNoSrcDecoderLayer(d_tar, d_hidden, n_heads=n_heads, droprate=droprate, \
				window=windows[i], attend_mode=attend_mode, gumbel_attend=gumbel_attend, use_wo=use_wo, \
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
