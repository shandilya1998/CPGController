import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

f = 'data/pretrain'
path = 'plots'

Y = np.load(os.path.join(f, 'Y.npy'))
MU = np.load(os.path.join(f, 'MEAN.npy'))
MEAN = np.load(os.path.join(f, 'MU.npy'))

num = Y.shape[0]
time = np.arange(Y.shape[1])

for j in tqdm(range(0, num, 6000)):
    y = Y[j]
    steps = y.shape[0]
    mean = MEAN[j]
    mu = MU[j]
    y = y * mu + mean
    fig, ax = plt.subplots(4,1, figsize = (5,20))
    for i in range(4):
        ax[i].plot(time, y[:, 3 * i], 'g', label = 'ankle')
        ax[i].plot(time, y[:, 3 * i + 1], 'b', label = 'knee')
        ax[i].plot(time, y[:, 3 * i + 2], 'r', label = 'hip')
        ax[i].legend()
    fig.savefig(os.path.join(f, path, 'plot_{j}.png'.format(j = j)))
    plt.close()

