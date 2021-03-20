import argparse
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--file', type = str, help = 'path of input file')
parser.add_argument('--out', type = str, help = 'path of output file')

args = parser.parse_args()

pkl = open(args.file, 'rb')
loss = pickle.load(pkl)
pkl.close()

if isinstance(loss[0], tf.Tensor):
    loss = [val.numpy() for val in loss]

fig, ax = plt.subplots(1,1, figsize = (5,5))
ax.plot(list(range(len(loss))), loss)
ax.set_xlabel('epochs')
ax.set_ylabel('loss')
fig.savefig(args.out)
