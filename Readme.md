[![Main Repo](https://img.shields.io/badge/Main_project-cotk-blue.svg?logo=github)](https://github.com/thu-coai/cotk)
[![This Repo](https://img.shields.io/badge/Model_repo-transformerLM--pytorch-blue.svg?logo=github)](https://github.com/thu-coai/transformerLM-pytorch)
[![Coverage Status](https://coveralls.io/repos/github/thu-coai/transformerLM-pytorch/badge.svg?branch=master)](https://coveralls.io/github/thu-coai/transformerLM-pytorch?branch=master)
[![Build Status](https://travis-ci.com/thu-coai/transformerLM-pytorch.svg?branch=master)](https://travis-ci.com/thu-coai/transformerLM-pytorch)

This repo is a benchmark model for [CoTK](https://github.com/thu-coai/cotk) package.

# transformerLM (PyTorch)

A transformer language model in pytorch.

You can refer to the following paper for details:

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

## Require Packages

* **python3**
* CoTK >= 0.1.0
* pytorch >= 1.0.0
* tensorboardX >= 1.4

## Quick Start

* Install ``CoTK`` following [instructions](https://github.com/thu-coai/cotk#installation).
* Using ``cotk download thu-coai/transformerLM-pytorch/master`` to download codes.
* Execute ``python run.py`` to train the model.
  * The default dataset is ``resources://MSCOCO``. You can use ``--dataid`` to specify data path (can be a local path, a url or a resources id). For example: ``--dataid /path/to/datasets``
  * It doesn't use pretrained word vector by default setting. You can use ``--wvid`` to specify data path for pretrained word vector (can be a local path, a url or a resources id). For example: ``--wvid resources://Glove300``
  * If you don't have GPUs, you can add `--cpu` for switching to CPU, but it may cost very long time for either training or test.
* You can view training process by tensorboard, the log is at `./tensorboard`.
  * For example, ``tensorboard --logdir=./tensorboard``. (You have to install tensorboard first.)
* After training, execute  ``python run.py --mode test --restore best`` for test.
  * You can use ``--restore filename`` to specify checkpoints files, which are in ``./model``. For example: ``--restore pretrained-mscoco`` for loading ``./model/pretrained-mscoco.model``
  * ``--restore last`` means last checkpoint, ``--restore best`` means best checkpoints on dev.
  * ``--restore NAME_last`` means last checkpoint with model named NAME. The same as``--restore NAME_best``.
* Find results at ``./output``.

## Arguments

```none
usage: run.py [-h] [--name NAME] [--restore RESTORE] [--mode MODE]
              [--pretrained_model PRETRAINED_MODEL]
              [--decode_mode {max,sample,gumbel,samplek,beam}] [--top_k TOP_K]
              [--length_penalty LENGTH_PENALTY] [--temperature TEMPERATURE]
              [--dataid DATAID]
              [--convert_to_lower_letter CONVERT_TO_LOWER_LETTER]
              [--epoch EPOCH] [--batch_per_epoch BATCH_PER_EPOCH]
              [--out_dir OUT_DIR] [--log_dir LOG_DIR] [--model_dir MODEL_DIR]
              [--cache_dir CACHE_DIR] [--cpu] [--debug] [--cache]
              [--seed SEED] [--lr LR]

A language generation model. Beamsearch, dropout and batchnorm is
supported.

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           The name of your model, used for tensorboard, etc.
                        Default: runXXXXXX_XXXXXX (initialized by current
                        time)
  --restore RESTORE     Checkpoints name to load. "NAME_last" for the last
                        checkpoint of model named NAME. "NAME_best" means the
                        best checkpoint. You can also use "last" and "best",
                        by default use last model you run. Attention:
                        "NAME_last" and "NAME_best" are not guaranteed to work
                        when 2 models with same name run in the same time.
                        "last" and "best" are not guaranteed to work when 2
                        models run in the same time. Default: None (don't load
                        anything)
  --mode MODE           "train" or "test". Default: train
  --pretrained_model PRETRAINED_MODEL
  --decode_mode {max,sample,gumbel,samplek,beam}
                        The decode strategy when freerun. Choices: max,
                        sample, gumbel(=sample), samplek(sample from topk),
                        beam(beamsearch). Default: samplek
  --top_k TOP_K         The top_k when decode_mode == "beam" or "samplek"
  --length_penalty LENGTH_PENALTY
                        The beamsearch penalty for short sentences. The
                        penalty will get larger when this becomes smaller.
  --temperature TEMPERATURE
                        Temperature. Default: 1
  --dataid DATAID       Directory for data set. Default: resources://MSCOCO
  --convert_to_lower_letter CONVERT_TO_LOWER_LETTER
                        Convert all tokens in dataset to lower case.
  --epoch EPOCH         Epoch for training. Default: 100
  --batch_per_epoch BATCH_PER_EPOCH
                        Batches per epoch. Default: 500
  --out_dir OUT_DIR     Output directory for test output. Default: ./output
  --log_dir LOG_DIR     Log directory for tensorboard. Default: ./tensorboard
  --model_dir MODEL_DIR
                        Checkpoints directory for model. Default: ./model
  --cache_dir CACHE_DIR
                        Checkpoints directory for cache. Default: ./cache
  --cpu                 Use cpu.
  --debug               Enter debug mode (using ptvsd).
  --cache               Use cache for speeding up load data and wordvec. (It
                        may cause problems when you switch dataset.)
  --seed SEED           Specify random seed. Default: 0
  --lr LR               Learning rate. Default: 0.0001
```

## Example

WAIT FOR UPDATE

## Performance

WAIT FOR UPDATE

## Author

[HUANG Fei](https://github.com/hzhwcmhf)
