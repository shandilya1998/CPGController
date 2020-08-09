import os
import numpy as np
import pickle
import re

folders = os.listdir('weights')

for folder in folders:
    files = os.listdir('weights/{folder}'.format(folder = folder))
    for f in files:
        path = os.path.join('weights', folder, f)
        pkl = open(path, 'rb')
        weights = pickle.load(pkl)
        pkl.close()
        path = os.path.join('weights', folder, f[:-6]+'npy')
        if os.path.exists(path):
            os.remove(path)
            os.listdir(os.path.join('weights', folder))
        np.save(path, weights)
