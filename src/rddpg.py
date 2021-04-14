import numpy as np
import tensorflow as tf
import os
import pickle
import time
import math
from rl.net import ActorNetwork, CriticNetwork
from rl.env import Env
from rl.constants import params

class Learner:
    def __init__(self, params):
        self.params = params


    def learn(self, model_dir, experiment):
        raise NotImplementedError

    def save(self, model_dir, exp):
        raise NotImplementedError
