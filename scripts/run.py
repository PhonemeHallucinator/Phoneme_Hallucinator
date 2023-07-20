import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import ast
import logging
import argparse
import numpy as np
import tensorflow as tf
from pprint import pformat

from utils.hparams import HParams
from models.runner import Runner
from models import get_model
from datasets import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--config', action='append', default=[])
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
params = HParams(args.cfg_file)
# parse config
for kv in args.config:
    k, v = kv.split('=', maxsplit=1)
    assert k, "Config item can't have empty key"
    assert v, "Config item can't have empty value"
    try:
        v = ast.literal_eval(v)
    except ValueError:
        v = str(v)
    params.update({k: v})
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=params.exp_dir + f'/{args.mode}.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)-15s %(message)s')
logging.info(pformat(params.dict))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(args.seed)
tf.set_random_seed(args.seed)
# data
trainset = get_dataset(params, 'train')
validset = get_dataset(params, 'valid')
testset = get_dataset(params, 'test')
logging.info((f"trainset: {trainset.size}",
              f"validset: {validset.size}",
              f"testset: {testset.size}"))

# model
model = get_model(params)
runner = Runner(params, model)
runner.set_dataset(trainset, validset, testset)

# run
if args.mode == 'train':
    runner.run()
elif args.mode == 'resume':
    model.load()
    runner.run()
elif args.mode == 'test':
    runner.evaluate()
else:
    raise ValueError()
