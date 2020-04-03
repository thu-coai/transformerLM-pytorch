#coding: utf-8
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.rnn import GRU
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .cuda_helper import zeros, Tensor, LongTensor, cuda, ones
from .gumbel import gumbel_max, gumbel_max_with_mask
from .storage import Storage

from transformers import GPT2LMHeadModel

class DecoderBase(nn.Module):
	def __init__(self):
		super().__init__()

	def _freerun(self, inp, max_sent_length, nextStep, updateEOS=None, mode='max', top_k=10, temperature=1):
		# inp contains: batch_size, dm, embLayer, max_sent_length, [init_h]
		# nextStep(embedding, flag):  pass embedding to RNN and get gru_h, flag indicates i th sentence is end when flag[i]==1

		# output: w_o emb length

		batch_size = inp.batch_size

		gen = Storage()
		gen.w_pro = []
		gen.w_o = []
		flag = zeros(batch_size).int()
		EOSmet = []

		next_emb = LongTensor([inp.go_id]).repeat(batch_size, 1)

		for i in range(max_sent_length):
			now = next_emb

			w = nextStep(now, flag, i) / temperature

			gen.w_pro.append(w.softmax(dim=-1))
			if mode == "max":
				w = torch.argmax(w, dim=1)
			elif mode == "gumbel" or mode == "sample":
				w_onehot = gumbel_max(w)[0]
				w = torch.argmax(w_onehot, dim=1)
			elif mode == "samplek":
				_, index = w.topk(top_k, dim=-1, largest=True, sorted=True) # batch_size, top_k
				mask = torch.zeros_like(w).scatter_(-1, index, 1.0)
				w_onehot = gumbel_max_with_mask(w, mask)
				w = torch.argmax(w_onehot, dim=1)

			gen.w_o.append(w)
			next_emb = w.unsqueeze(-1)

			EOSmet.append(flag)
			flag = flag | (w == inp.eos_id).int()
			if updateEOS:
				flag = updateEOS(flag)

			if (i+1) % 10 == 0:
				if torch.sum(flag).detach().cpu().numpy() == batch_size:
					break

		EOSmet = 1 - torch.stack(EOSmet)
		gen.length = torch.sum(EOSmet.long(), 0).detach().cpu().numpy()
		seqlen = max(gen.length)
		EOSmet = EOSmet[:seqlen]
		gen.w_o = (torch.stack(gen.w_o[:seqlen]) * EOSmet.long()).transpose(0, 1)

		return gen

	def _beamsearch(self, inp, max_length, nextStep, top_k=10, length_penalty=0.7, temperature=1):

		batch_size = inp.batch_size

		w_o = []
		flag = zeros(batch_size, top_k).byte()
		EOSmet = []
		score = zeros(batch_size, top_k)
		score[:, 1:] = -1e8
		now_length = zeros(batch_size, top_k)
		back_index = []
		regroup = LongTensor([i for i in range(top_k)]).repeat(batch_size, 1)

		next_emb = LongTensor([inp.go_id]).repeat(batch_size, top_k)

		for i in range(max_length):
			now = next_emb

			w, valid_flag = nextStep(now, flag, i, regroup=regroup) # batch_size, top_k, hidden_size
			w = w / temperature
			valid_flag = (1 - flag.float()) * valid_flag.float()

			w = w.log_softmax(dim=-1)

			new_score = (score.unsqueeze(-1) + w * valid_flag.unsqueeze(-1)) / ((now_length.float() + valid_flag + 1e-9).unsqueeze(-1) ** length_penalty)
			new_score[:, :, 1:] = new_score[:, :, 1:] + (1 - valid_flag.float()).unsqueeze(-1) * new_score.min() * 10
			_, index = new_score.reshape(batch_size, -1).topk(top_k, dim=-1, largest=True, sorted=True) # batch_size, top_k

			sum_score = (score.unsqueeze(-1) + w * valid_flag.unsqueeze(-1)).reshape(batch_size, -1)
			# assert (regroup >= new_score.shape[1]).sum().tolist() == 0
			score = torch.gather(sum_score, dim=1, index=index)

			vocab_size = w.shape[-1]
			regroup = index / vocab_size # batch_size, top_k
			back_index.append(regroup)
			w_id = torch.fmod(index, vocab_size) # batch_size, top_k

			# regroup = cuda(torch.LongTensor(np.arange(10))).unsqueeze(0).repeat(32, 1)
			# _, w = w.max(dim=-1)

			# assert (regroup >= flag.shape[1]).sum().tolist() == 0
			flag = torch.gather(flag, dim=1, index=regroup)
			# assert (regroup >= now_length.shape[1]).sum().tolist() == 0
			now_length = torch.gather(now_length, dim=1, index=regroup) + valid_flag

			w_o.append(w_id)
			next_emb = w_id

			EOSmet.append(flag)

			flag = flag | ((w_id == inp.eos_id) * valid_flag).byte()

			if torch.sum(flag).detach().cpu().numpy() == batch_size * top_k:
				break

		#back tracking
		gen = Storage()
		back_EOSmet = []
		gen.w_o = []
		now_index = LongTensor([i for i in range(top_k)]).repeat(batch_size, 1)

		for i, index in reversed(list(enumerate(back_index))):
			gen.w_o.append(torch.gather(w_o[i], dim=1, index=now_index))
			back_EOSmet.append(torch.gather(EOSmet[i], dim=1, index=now_index))
			now_index = torch.gather(index, dim=1, index=now_index)

		back_EOSmet = 1-torch.stack(list(reversed(back_EOSmet)))
		gen.w_o = torch.stack(list(reversed(gen.w_o))) * back_EOSmet.long()
		gen.length = torch.sum(back_EOSmet, 0).detach().cpu().numpy()

		return gen


class GPT2Decoder(DecoderBase):
	def __init__(self, pretrained_model):
		super().__init__()
		self.model = GPT2LMHeadModel.from_pretrained(pretrained_model)

	def forward(self, incoming, attn_mask):
		logits, _ = self.model(incoming, attention_mask=attn_mask)
		return logits

	def init_forward(self):
		past = None

		def nextStep(incoming, stopmask, position):
			nonlocal past
			logits, present = self.model(input_ids=incoming, attention_mask=1-stopmask, past=past)
			past = present
			return logits[:, 0, :]

		return nextStep

	def init_forward_3d(self, batch_size, top_k):
		past = None
		i = 0

		def nextStep(incoming, stopmask, position, regroup=None):
			nonlocal past, i, top_k
			i = position

			new_incoming = incoming.unsqueeze(-1)


			flatten_batch = batch_size * top_k
			new_incoming = new_incoming.reshape(flatten_batch, -1)
			stopmask = stopmask.reshape(flatten_batch)

			if past:
				d1, d2, d3, d4, d5 = past[0].shape
				assert d2 == flatten_batch
				past = tuple(m.reshape(d1, batch_size, top_k, d3, d4, d5).\
						gather(2, index=regroup.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(d1, 1, 1, d3, d4, d5)).\
						reshape(d1, flatten_batch, d3, d4, d5) for m in past)

			logits, present = self.model(input_ids=new_incoming, attention_mask=1-stopmask.unsqueeze(-1), \
				past=past)

			past = present

			valid_flag = ones(batch_size, top_k)
			return logits[:, 0, :].reshape(batch_size, top_k, -1), valid_flag

		return nextStep

	def freerun(self, inp, mode='max', top_k=10, temperature=1):
		nextStep = self.init_forward()
		return self._freerun(inp, inp.max_sent_length, nextStep, None, mode, top_k, temperature)

	# def beamsearch(self, inp, top_k, wLinearLayerCallback, input_callback=None, length_penalty=0.7):
	# 	nextStep = self.init_forward_3d(inp.batch_size, top_k, inp.get("init_h", None))
	# 	return self._beamsearch(inp, top_k, nextStep, wLinearLayerCallback, input_callback, length_penalty)



class GPT2Seq2seq(DecoderBase):
	def __init__(self, pretrained_model):
		super().__init__()
		self.model = GPT2LMHeadModel.from_pretrained(pretrained_model)

	def _concat(self, post, post_length, resp, resp_length, go_id):
		batch_size = len(post)
		concat_length = [post_length[i] + resp_length[i] - 1 for i in range(batch_size)]
		indexes = np.zeros((batch_size, max(concat_length)), dtype=int)
		incoming = ones(batch_size, max(concat_length), device=post).long() * go_id
		attn_mask = zeros(batch_size, max(concat_length), device=post)
		type_mask = zeros(batch_size, max(concat_length), device=post).long()

		for i in range(batch_size):
			indexes[i, :post_length[i]] = np.arange(post_length[i])
			indexes[i, post_length[i]:concat_length[i]] = np.arange(resp_length[i] - 1) + post.shape[1] + 1
			attn_mask[i, :concat_length[i]] = 1
			type_mask[i, post_length[i]:concat_length[i]] = 1
		incoming = torch.cat([post, resp], 1).gather(1, LongTensor(indexes))

		return incoming, attn_mask, type_mask

	def _split(self, incoming, post_length, resp_length):
		batch_size = len(incoming)
		indexes = np.zeros((batch_size, max(resp_length)), dtype=int)
		for i in range(batch_size):
			indexes[i, :resp_length[i]-1] = np.arange(resp_length[i] - 1) + post_length[i] - 1
			indexes[i, resp_length[i]-1:] = resp_length[i] - 1 + post_length[i] - 1

		if len(incoming.shape) == 2:
			res = torch.gather(incoming, 1, LongTensor(indexes))
		else:
			res = torch.gather(incoming, 1, LongTensor(indexes).unsqueeze(-1).expand(-1, -1, incoming.shape[-1]))
		return res

	def forward(self, post, post_length, resp, resp_length, go_id):
		incoming, attn_mask, type_mask = self._concat(post, post_length, resp, resp_length, go_id)
		logits, _ = self.model(incoming, attention_mask=attn_mask, token_type_ids=type_mask)
		return self._split(logits, post_length, resp_length)

	def init_forward(self, batch_size, post, post_length):
		past = None
		i = 0
		type_mask = ones(batch_size, post.shape[1], device=post).long()
		for i in range(batch_size):
			type_mask[i, :post_length[i]] = 0

		def nextStep(incoming, stopmask, position):
			nonlocal past, i
			i = position

			if i < post.shape[1]:
				new_incoming = post[:, i:i+1] * (1 - type_mask[:, i:i+1]) + incoming * type_mask[:, i:i+1]
				new_type_mask = type_mask[:, i:i+1]
			else:
				new_incoming = incoming
				new_type_mask = torch.ones_like(type_mask[:, :1])

			logits, present = self.model(input_ids=new_incoming, attention_mask=1-stopmask.unsqueeze(-1), \
				token_type_ids=new_type_mask, past=past)
			past = present
			return logits[:, 0, :]

		def updateStopMask(stopmask):
			nonlocal i
			if i < post.shape[1]:
				return type_mask[:, i].int() * stopmask
			else:
				return stopmask

		return nextStep, updateStopMask

	def init_forward_3d(self, batch_size, post, post_length, top_k):
		past = None
		i = 0
		type_mask = ones(batch_size, post.shape[1], device=post).long()
		for i in range(batch_size):
			type_mask[i, :post_length[i]] = 0
		type_mask = type_mask.unsqueeze(1).expand(batch_size, top_k, post.shape[1])

		def nextStep(incoming, stopmask, position, regroup=None):
			nonlocal past, i, post
			i = position

			if i < post.shape[1]:
				post_3d = post[:, i:i+1].unsqueeze(1).expand(batch_size, top_k, 1)
				new_incoming = post_3d * (1 - type_mask[:, :, i:i+1]) + incoming.unsqueeze(-1) * type_mask[:, :, i:i+1]
				new_type_mask = type_mask[:, :, i:i+1]
			else:
				new_incoming = incoming.unsqueeze(-1)
				new_type_mask = torch.ones_like(type_mask[:, :, :1])


			flatten_batch = batch_size * top_k
			new_incoming = new_incoming.reshape(flatten_batch, -1)
			new_type_mask = new_type_mask.reshape(flatten_batch, -1)
			stopmask = stopmask.reshape(flatten_batch)

			if past:
				d1, d2, d3, d4, d5 = past[0].shape
				assert d2 == flatten_batch
				past = tuple(m.reshape(d1, batch_size, top_k, d3, d4, d5).\
						gather(2, index=regroup.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(d1, 1, 1, d3, d4, d5)).\
						reshape(d1, flatten_batch, d3, d4, d5) for m in past)

			logits, present = self.model(input_ids=new_incoming, attention_mask=1-stopmask.unsqueeze(-1), \
				token_type_ids=new_type_mask, past=past)

			past = present

			if i + 1 < post.shape[1]:
				valid_flag = type_mask[:, :, i + 1].int()
			else:
				valid_flag = torch.ones_like(type_mask[:, :, 0])
			return logits[:, 0, :].reshape(batch_size, top_k, -1), valid_flag

		return nextStep

	def freerun(self, inp, mode='max', top_k=10, temperature=1):
		nextStep, updateStopMask = self.init_forward(inp.batch_size, inp.post, inp.post_length)
		gen = self._freerun(inp, inp.max_sent_length + inp.post.shape[1], nextStep, updateStopMask, mode, top_k, temperature)
		res = Storage()
		res.length = np.minimum(gen.length - inp.post_length, np.array([inp.max_sent_length - 1]))
		res.w_o = self._split(gen.w_o, inp.post_length, res.length + 1)
		return res

	def beamsearch(self, inp, top_k, length_penalty=0.7, temperature=1):
		nextStep = self.init_forward_3d(inp.batch_size, inp.post, inp.post_length, top_k)
		gen = self._beamsearch(inp, inp.max_sent_length + inp.post.shape[1], nextStep, top_k, length_penalty, temperature)
		res = Storage()
		res.length = np.minimum(gen.length[:, 0] - inp.post_length, np.array([inp.max_sent_length - 1]))
		res.w_o = self._split(gen.w_o[:, :, 0].transpose(0, 1), inp.post_length, res.length + 1)
		return res
