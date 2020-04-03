# coding:utf-8

def run(*argv):
	import argparse
	import time

	from utils import Storage
	from pathlib import Path

	parser = argparse.ArgumentParser(description='A language model with transformer decoder.')
	args = Storage()

	parser.add_argument('--name', type=str, default=None,
		help='The name of your model, used for tensorboard, etc. Default: runXXXXXX_XXXXXX (initialized by current time)')
	parser.add_argument('--restore', type=str, default=None,
		help='Checkpoints name to load. \
			"NAME_last" for the last checkpoint of model named NAME. "NAME_best" means the best checkpoint. \
			You can also use "last" and "best", by default use last model you run. \
			Attention: "NAME_last" and "NAME_best" are not guaranteed to work when 2 models with same name run in the same time. \
			"last" and "best" are not guaranteed to work when 2 models run in the same time.\
			Default: None (don\'t load anything)')
	parser.add_argument('--mode', type=str, default="train",
		help='"train" or "test". Default: train')

	parser.add_argument('--tf_size', type=int, default=256,
		help='feature size of transformer')
	parser.add_argument('--tf_hidden_size', type=int, default=512,
		help='hidden size of mlp in transformer')
	parser.add_argument('--n_heads', type=int, default=4,
		help='number of heads in transformer')
	parser.add_argument('--n_layers', type=int, default=5,
		help='number of layers in transformer')
	parser.add_argument('--input_droprate', type=float, default=0.1,
		help='the droprate(the probability to be zeroed) of input embedding in transformer. 0 indicates for don\'t use dropout')
	parser.add_argument('--droprate', type=float, default=0.25,
		help='The probability to be zeroed. 0 indicates for don\'t use dropout')
	parser.add_argument('--batch_size', type=int, default=256,
		help='number of sample in a batch')

	parser.add_argument('--decode_mode', type=str, choices=['max', 'sample', 'gumbel', 'samplek', 'beam'], default='samplek',
		help='The decode strategy when freerun. Choices: max, sample, gumbel(=sample), \
			samplek(sample from topk), beam(beamsearch). Default: samplek')
	parser.add_argument('--top_k', type=int, default=10,
		help='The top_k when decode_mode == "beam" or "samplek"')
	parser.add_argument('--length_penalty', type=float, default=0.7,
		help='The beamsearch penalty for short sentences. The penalty will get larger when this becomes smaller.')
	parser.add_argument('--temperature', type=float, default=1)

	parser.add_argument('--dataid', type=str, default='resources://MSCOCO',
		help='Directory for data set. Default: resources://MSCOCO')
	parser.add_argument('--epoch', type=int, default=100,
		help="Epoch for training. Default: 100")
	parser.add_argument('--batch_per_epoch', type=int, default=500,
		help="Batches per epoch. Default: 500")
	parser.add_argument('--wvid', type=str, default="resources://Glove300d",
		help="Directory for pretrained wordvector. Default: resources://Glove300d")

	parser.add_argument('--out_dir', type=str, default="./output",
		help='Output directory for test output. Default: ./output')
	parser.add_argument('--log_dir', type=str, default="./tensorboard",
		help='Log directory for tensorboard. Default: ./tensorboard')
	parser.add_argument('--model_dir', type=str, default="./model",
		help='Checkpoints directory for model. Default: ./model')
	parser.add_argument('--cache_dir', type=str, default="./cache",
		help='Checkpoints directory for cache. Default: ./cache')
	parser.add_argument('--cpu', action="store_true",
		help='Use cpu.')
	parser.add_argument('--debug', action='store_true',
		help='Enter debug mode (using ptvsd).')
	parser.add_argument('--cache', action='store_true',
		help='Use cache for speeding up load data and wordvec. (It may cause problems when you switch dataset.)')
	parser.add_argument('--seed', type=int, default=0,
		help='Specify random seed. Default: 0')
	parser.add_argument('--lr', type=float, default=1e-3,
		help='Learning rate. Default: 0.001')
	cargs = parser.parse_args(argv)


	# general setting
	args.name = cargs.name or time.strftime("run%Y%m%d_%H%M%S", time.localtime())
	args.restore = cargs.restore
	args.mode = cargs.mode
	args.out_dir = cargs.out_dir
	args.log_dir = cargs.log_dir
	args.model_dir = cargs.model_dir
	args.cache_dir = cargs.cache_dir
	args.debug = cargs.debug
	args.cache = cargs.cache
	args.cuda = not cargs.cpu

	## dataset settings
	args.dataid = cargs.dataid
	args.tokenizer = "space"
	args.max_sent_length = 50
	args.convert_to_lower_letter = False
	args.min_frequent_vocab_times = 10
	args.min_rare_vocab_times = 0
	args.wvid = cargs.wvid

	## training settings
	args.epochs = cargs.epoch
	args.lr = cargs.lr
	args.batch_size = 128
	args.batch_num_per_gradient = 1
	args.grad_clip = 5
	args.show_sample = [0]  # show which batch when evaluating at tensorboard
	args.checkpoint_steps = 20
	args.checkpoint_max_to_keep = 5

	## arguments for restoring checkpoints
	args.restore_optimizer = True
	load_exclude_set = []
	restoreCallback = None

	## architecture settings
	args.batch_per_epoch = cargs.batch_per_epoch
	args.embedding_size = 300
	args.tf_size = cargs.tf_size
	args.tf_hidden_size = cargs.tf_hidden_size
	args.n_heads = cargs.n_heads
	args.n_layers = cargs.n_layers
	args.input_droprate = cargs.input_droprate
	args.droprate = cargs.droprate

	## decoding settings
	args.decode_mode = cargs.decode_mode
	args.top_k = cargs.top_k
	args.length_penalty = cargs.length_penalty
	args.temperature = cargs.temperature

	## random seed
	args.seed = cargs.seed

	import random
	random.seed(cargs.seed)
	import torch
	torch.manual_seed(cargs.seed)
	import numpy as np
	np.random.seed(cargs.seed)

	from main import main

	return main(args, load_exclude_set, restoreCallback)

if __name__ == '__main__':
	import sys
	run(*sys.argv[1:])
