from rl.torch.env import Env
from rl.torch.rdpg import RDPG
from rl.torch.util import *
from rl.torch.constants import params
import numpy as np
import argparse
from copy import deepcopy
import torch
from torchsummary import summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type = int,
        help = 'ID of experiment being performaed'
    )
    parser.add_argument(
        '--out_path',
        type = str,
        help = 'Path to output directory'
    )
    args = parser.parse_args()
    env = Env(
        params,
        args.experiment
    )
    rdpg = RDPG(env, params)
    checkpoint_path = os.path.join(args.out_path, 'exp{exp}'.format(
        exp = args.experiment
    ))
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    with torch.autograd.set_detect_anomaly(True):
        rdpg.train(params['train_episode_count'], checkpoint_path, True)

