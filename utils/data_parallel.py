# copy from pytorch 1.2.0

import operator
import torch
import warnings
import threading
from itertools import chain
from torch.nn.modules import Module
from torch.nn.parallel.replicate import replicate
from torch.cuda._utils import _get_device_index

def get_a_var(obj):
	if isinstance(obj, torch.Tensor):
		return obj

	if isinstance(obj, list) or isinstance(obj, tuple):
		for result in map(get_a_var, obj):
			if isinstance(result, torch.Tensor):
				return result
	if isinstance(obj, dict):
		for result in map(get_a_var, obj.items()):
			if isinstance(result, torch.Tensor):
				return result
	return None

def parallel_apply(modules, funcname, inputs, kwargs_tup=None, devices=None):
	r"""Applies each `module` in :attr:`modules` in parallel on arguments
	contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
	on each of :attr:`devices`.

	Args:
		modules (Module): modules to be parallelized
		inputs (tensor): inputs to the modules
		devices (list of int or torch.device): CUDA devices

	:attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
	:attr:`devices` (if given) should all have same length. Moreover, each
	element of :attr:`inputs` can either be a single object as the only argument
	to a module, or a collection of positional arguments.
	"""
	assert len(modules) == len(inputs)
	if kwargs_tup is not None:
		assert len(modules) == len(kwargs_tup)
	else:
		kwargs_tup = ({},) * len(modules)
	if devices is not None:
		assert len(modules) == len(devices)
	else:
		devices = [None] * len(modules)
	devices = list(map(lambda x: _get_device_index(x, True), devices))
	lock = threading.Lock()
	results = {}
	grad_enabled = torch.is_grad_enabled()

	def _worker(i, module, input, kwargs, device=None):
		torch.set_grad_enabled(grad_enabled)
		if device is None:
			device = get_a_var(input).get_device()
		try:
			with torch.cuda.device(device):
				# this also avoids accidental slicing of `input` if it is a Tensor
				if not isinstance(input, (list, tuple)):
					input = (input,)
				output = getattr(module, funcname)(*input, **kwargs)
			with lock:
				results[i] = output
		except Exception as e:
			with lock:
				results[i] = e

	if len(modules) > 1:
		threads = [threading.Thread(target=_worker,
									args=(i, module, input, kwargs, device))
				   for i, (module, input, kwargs, device) in
				   enumerate(zip(modules, inputs, kwargs_tup, devices))]

		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()
	else:
		_worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

	outputs = []
	for i in range(len(inputs)):
		output = results[i]
		if isinstance(output, Exception):
			raise output
		outputs.append(output)
	return outputs

def _check_balance(device_ids):
	imbalance_warn = """
	There is an imbalance between your GPUs. You may want to exclude GPU {} which
	has less than 75% of the memory or cores of GPU {}. You can do so by setting
	the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
	environment variable."""
	device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
	dev_props = [torch.cuda.get_device_properties(i) for i in device_ids]

	def warn_imbalance(get_prop):
		values = [get_prop(props) for props in dev_props]
		min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
		max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
		if min_val / max_val < 0.75:
			warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
			return True
		return False

	if warn_imbalance(lambda props: props.total_memory):
		return
	if warn_imbalance(lambda props: props.multi_processor_count):
		return

class MyDataParallel(Module):
	r"""Implements data parallelism at the module level.

	Args:
		module (Module): module to be parallelized
		device_ids (list of int or torch.device): CUDA devices (default: all devices)
		output_device (int or torch.device): device location of output (default: device_ids[0])

	Attributes:
		module (Module): the module to be parallelized

	Example::

		>>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
		>>> output = net(input_var)  # input_var can be on any device, including CPU
	"""

	# TODO: update notes/cuda.rst when this class handles 8+ GPUs well

	def __init__(self, module, device_ids=None, output_device=None):
		super().__init__()

		if not torch.cuda.is_available():
			self.module = module
			self.device_ids = []
			return

		if device_ids is None:
			device_ids = list(range(torch.cuda.device_count()))
		if output_device is None:
			output_device = device_ids[0]

		self.module = module
		self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
		self.output_device = _get_device_index(output_device, True)
		self.src_device_obj = torch.device("cuda:{}".format(self.device_ids[0]))

		_check_balance(self.device_ids)

		if len(self.device_ids) == 1:
			self.module.cuda(device_ids[0])

	def apply_parallel(self, funcname):

		def forward(inputs, kwargs=None):
			if not self.device_ids:
				if (isinstance(inputs, list) or isinstance(inputs, tuple)):
					if len(inputs) > 1:
						raise RuntimeError("the length of inputs must be equal to the length of device_ids")
					return getattr(self.module, funcname)(*inputs[0], **kwargs[0])
				else:
					return getattr(self.module, funcname)(*inputs, **kwargs)

			if len(self.device_ids) != len(inputs):
				raise RuntimeError("the length of inputs must be equal to the length of device_ids")

			for t in chain(self.module.parameters(), self.module.buffers()):
				if t.device != self.src_device_obj:
					raise RuntimeError("module must have its parameters and buffers "
									"on device {} (device_ids[0]) but found one of "
									"them on device: {}".format(self.src_device_obj, t.device))

			if len(self.device_ids) == 1:
				return getattr(self.module, funcname)(*inputs[0], **kwargs[0])
			replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
			outputs = self.parallel_apply(replicas, funcname, inputs, kwargs)
			return outputs

		return forward

	def replicate(self, module, device_ids):
		return replicate(module, device_ids, not torch.is_grad_enabled())

	def parallel_apply(self, replicas, funcname, inputs, kwargs):
		return parallel_apply(replicas, funcname, inputs, kwargs, self.device_ids[:len(replicas)])
