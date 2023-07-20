import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import ast
import logging
import argparse
import pickle
import numpy as np
import tensorflow as tf
from pprint import pformat
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import tensorflow as tf

from utils.hparams import HParams
from models import get_model
from datasets.speech import Dataset
from sklearn.manifold import TSNE
import seaborn as sns
clrs = sns.color_palette("husl", 5)


parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
params = HParams(args.cfg_file)
# modify config
params.mask_type = 'det_expand'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#np.random.seed(args.seed)
#tf.set_random_seed(args.seed)

# data
testset = Dataset("test", batch_size=100, set_size=200, mask_type=params.mask_type)
testset.initialize()

# model
model = get_model(params)
model.load()

# run
save_dir = f'{params.exp_dir}/evaluate/speech_expansion/'
os.makedirs(save_dir, exist_ok=True)
log_file = open(f'{save_dir}/log.txt', 'w')

def evaluate(batch):
    sample = model.execute(model.sample, batch)
    return sample

def visualize(input, mask, sample, save_path):
    
    #with open(f'{save_path}data.pkl', 'wb') as f:
    #    pickle.dump((batch, sample), f)
    sample = sample
    mask = mask
    input = input
    expanded = input * mask + sample * (1 - mask)
  
    N = sample.shape[0]
    D = sample.shape[1]
    C = sample.shape[2]
    if True:
        cdict = {1: 'red', 2: 'blue', 3: 'green'}
        for i in range(N):
            expanded_embedded = TSNE(n_components=2, learning_rate=100, init='random', perplexity=60).fit_transform(expanded[i])
 
            plt.figure(figsize=(9,9))
            plt.tight_layout()
            idx = np.where(mask[i,:,0]==0)
            plt.scatter(expanded_embedded[idx, 0], expanded_embedded[idx,1], c = "#1A85FF", label="Synthesized", alpha=0.2)
            idx = np.where(mask[i,:,0]==1)
            plt.scatter(expanded_embedded[idx, 0], expanded_embedded[idx,1], c = "#D41159", label="Real")
            plt.legend(fontsize="20", loc ="upper left")
            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
            plt.savefig(f"{save_path}embed_{i}.png", dpi=200)
            plt.close()
    if False:
        D_id_color = {'0': u'orchid', '1': u'darkcyan', '2': u'grey', '3': u'dodgerblue', '4': u'turquoise', '5': u'darkviolet'}
        sample = sample.reshape(N * D, C)
        mask = mask.reshape(N * D, C)
        input = input.reshape(N * D, C)
        expanded = expanded.reshape(N * D, C)
        expanded_embedded = TSNE(n_components=2, learning_rate=1, init='random', perplexity=50).fit_transform(expanded)
        expanded_embedded = expanded_embedded.reshape(N, D, 2)
        plt.figure(figsize=(10,10))
        plt.tight_layout()
        for i in range(N):
            plt.scatter(expanded_embedded[i, :, 0], expanded_embedded[i, :,1], c = D_id_color[str(i % 6)])
        plt.savefig(f"{save_path}embed.png", dpi=200)
        plt.close()
    
    


# test
save_path = f'{save_dir}/test/'
os.makedirs(save_path, exist_ok=True)
samples = []
inputs = []
masks = []
filenames = []
num_sample_step = 20

batch = testset.next_batch()
for s in range(num_sample_step):
    sample = evaluate(batch)
    samples.append(sample)
samples = np.concatenate(samples, axis=1)
inputs = np.tile(batch['x'], (1, num_sample_step, 1))
masks = np.tile(batch['b'], (1, num_sample_step, 1))
visualize(inputs, masks, samples, save_path)
log_file.close()