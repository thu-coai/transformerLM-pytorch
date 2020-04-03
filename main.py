# coding:utf-8
import os
import logging
import json

from cotk.dataloader import LanguageGeneration
from cotk.wordvector import GeneralWordVector
from utils import debug, try_cache, cuda_init, Storage
from transformerlm import TransformerLM

def main(args, load_exclude_set, restoreCallback):
	logging.basicConfig(\
		filename=0,\
		level=logging.DEBUG,\
		format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',\
		datefmt='%H:%M:%S')

	if args.debug:
		debug()
	logging.info(json.dumps(args, indent=2))

	cuda_init(0, args.cuda)

	volatile = Storage()
	volatile.load_exclude_set = load_exclude_set
	volatile.restoreCallback = restoreCallback

	data_class = LanguageGeneration
	data_arg = Storage()
	data_arg.file_id = args.dataid
	data_arg.tokenizer = args.tokenizer
	data_arg.max_sent_length = args.max_sent_length
	data_arg.convert_to_lower_letter = args.convert_to_lower_letter
	data_arg.min_frequent_vocab_times = args.min_frequent_vocab_times
	data_arg.min_rare_vocab_times = args.min_rare_vocab_times
	wordvec_class = GeneralWordVector

	def load_dataset(data_arg, wvpath, embedding_size):
		wv = wordvec_class(wvpath)
		dm = data_class(**data_arg)
		return dm, wv.load_matrix(embedding_size, dm.frequent_vocab_list)

	if args.cache:
		dm, volatile.wordvec = try_cache(load_dataset, (data_arg, args.wvpath, args.embedding_size),
			args.cache_dir, data_class.__name__ + "_" + wordvec_class.__name__)
	else:
		dm, volatile.wordvec = load_dataset(data_arg, args.wvpath, args.embedding_size)

	volatile.dm = dm

	param = Storage()
	param.args = args
	param.volatile = volatile

	model = TransformerLM(param)
	if args.mode == "train":
		model.train_process()
	elif args.mode == "test":
		test_res = model.test_process()

		json.dump(test_res, open("./result.json", "w"))
	elif args.mode == "load":
		return model
	else:
		raise ValueError("Unknown mode")
