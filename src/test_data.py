import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

f = 'data/pretrain'
path = 'plots'

Y = np.load(os.path.join(f, 'Y.npy'))
MU = np.load(os.path.join(f, 'MEAN.npy'))
MEAN = np.load(os.path.join(f, 'MU.npy'))

steps = Y.shape[1]

y = Y * np.repeat(np.expand_dims(MU * np.pi/3, 1), steps, 1) + np.repeat(np.expand_dims(MEAN, 1), steps, 1)
num = Y.shape[0]

for j in tqdm(range(0, num, 1000)):
    fig, ax = plt.subplots(4,1, figsize = (5,20))
    for i in range(4):
        ax[i].plot(y[j][:, 3 * i], 'g', label = 'ankle')
        ax[i].plot(y[j][:, 3 * i + 1], 'b', label = 'knee')
        ax[i].plot(y[j][:, 3 * i + 2], 'r', label = 'hip')
        ax[i].legend()
    fig.savefig(os.path.join(f, path, 'plot_{j}.png'.format(j = j)))
    plt.close()

