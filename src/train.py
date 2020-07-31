import numpy as np
import gait_generation as gg
from frequency_analysis import frequency_estimator
from tqdm import tqdm
from model1 import OutputMLP

class DataLoader(object):
    def __init__(self, 
                 dt, 
                 N):
        self.dt = dt
        self.N = N

    def get_signal(self, Tsw, Tst, theta, )

class Train(object):
    def __init__(self, 
                 dt, 
                 N, 
                 nepcohs,
                 num_osc, 
                 num_h, 
                 num_out, 
                 init = 'random'):
        self.num_osc = num_osc
        self.num_h = num_h
        self.num_out = num_out
        self.init = init
        self.dt = dt
        self.N = N
        self.nepochs
        self.out_mlp = OutputMLP()
        
    def __call__(self):
        self.out_mlp.build(self.num_osc, 
                           self.num_h, 
                           self.num_out, self.init)
        for i in tqdm(range(self.nepochs)):
            
