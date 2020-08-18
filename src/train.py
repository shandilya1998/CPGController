import numpy as np
from gait_generation import gait_generator as gg
from frequency_analysis import frequency_estimator
from tqdm import tqdm
from layers.OutputMLP import OutputMLP
from layers.OscLayer import OscLayer
import pickle
import matplotlib.pyplot as plt
import math
import scipy.fft as fft
import scipy.signal as signal 

class DataLoader(object):
    def __init__(self, 
                 dt, 
                 N,
                 num_osc):
        self.dt = dt
        self.N = N
        self.num_osc = num_osc
        self.osc = OscLayer(self.num_osc, self.N, self.dt)
    
    def preprocess(self):
        self.signal[:, 1:] = self.signal[:, 1:]-np.mean(self.signal[:, 1:],
                                                        axis = 0)
        self.signal[:, 1:] = self.signal[:, 1:]/(1.2*np.abs(self.signal[:, 1:].max(axis = 0)))
    
    def set_signal(self, Tsw, Tst, theta):
        self.signal =  gg.get_signal(self.dt, Tsw, Tst, self.N, theta)
        self.preprocess()

    def get_ff(self, signal, ff_type = 'fft'):
        if ff_type == 'fft': 
            return frequency_estimator.freq_from_fft(signal, 1/self.dt)
        elif ff_type == 'autocorr':
            return frequency_estimator.freq_from_autocorr(signal, 1/self.dt)    
    
    def set_num_osc(self, num):
        self.num_osc = num
  
    def get_input(self, Tsw, Tst, theta):
        Z = np.empty((self.num_osc, N), dtype = np.complex128)
        self.set_signal(Tsw, Tst, theta)
        """
            Assuming a straight line motion with a constant speed
        """
        ff = self.get_ff(self.signal[:, 1], 'autocorr')
        freq = np.empty((self.num_osc,))
        for i in range(self.num_osc):
            freq[i] = 2*np.pi*ff*(i+1)
        print(freq)
        return self.osc.get_signal(freq)
                
class Train(object):
    def __init__(self, 
                 dt, 
                 N, 
                 nepochs,
                 num_osc, 
                 num_h, 
                 num_out, 
                 exp,
                 init = 'random',
                 lr = 1e-3):
        self.num_osc = num_osc
        self.num_h = num_h
        self.num_out = num_out
        self.init = init
        self.dt = dt
        self.N = N
        self.nepochs = nepochs
        self.data = DataLoader(self.dt, self.N, self.num_osc)
        self.out_mlp = OutputMLP()
        self.lr = lr
        self.err = np.zeros((self.nepochs,))
        self.exp = exp

    def _plot(self, axis, x, y):
        axis.plot(
            self.data.signal[:500, 0].T, 
            x[:500],
            'r',
            label = 'ideal gait')
        axis.plot(
            self.data.signal[:500, 0].T, 
            y[:500], 
            'b',
            label = 'generated gait')
        axis.set_xlabel('time')
        axis.set_ylabel('joint activation')
        axis.legend()    

    def plot(self, yr):
        fig, axes = plt.subplots(self.num_out, 1, figsize = (5, 5*self.num_out))
        if self.num_out!=1:
            for i in range(self.num_out):
                self._plot(axes[i], self.data.signal[:, i+1].T, yr[i])
        else:
            self._plot(axes, self.data.signal[:, 1].T, yr[0])
        fig.savefig('../images/pred_vs_ideal_exp{exp}.png'.format(exp=self.exp))
        plt.show()

    def __call__(self):
        self.out_mlp.build(self.num_osc, 
                           self.num_h, 
                           self.num_out,
                           0.5, 
                           self.init)
        """
            Assuming a straight line motion with a constant speed
        """
        Tst = 60
        Tsw = 20
        theta = 15
        Z = self.data.get_input(Tsw, Tst, theta)
        for i in tqdm(range(self.nepochs)):
            yr = self.out_mlp(Z, self.out_mlp.sigmoidf)
            err = np.sum((self.data.signal[:, 1:self.num_out+1].T-yr)**2)
            #print(err)
            self.err[i] = err            
            """
                Back propagation
            """
            self.out_mlp.backprop_sigmoid(yr, Z, self.data, self.lr)
            if i>20 and (math.isclose(
                self.err[i], 
                self.err[i-20], 
                rel_tol = 1e-5, 
                abs_tol=1e-5,
            ) or self.err[i]>self.err[i-20]):
                self.lr = self.lr*0.5
                #break 
                #"""
                if math.isclose(
                    self.lr,
                    1e-15,
                    abs_tol = 1e-16,
                    rel_tol = 1e-16
                ):  
                    self.lr = 1e-3
                #"""    
        
        yr = self.out_mlp(Z, self.out_mlp.sigmoidf)
        np.save(
            'weights/exp{exp}/w2_out_mlp.npy'.format(exp=self.exp), 
            self.out_mlp.W2
        )
        np.save(
            'weights/exp{exp}/w1_out_mlp.npy'.format(exp=self.exp), 
            self.out_mlp.W1
        ) 
        fig, axes = plt.subplots(1, 1, figsize = (5, 5))
        axes.plot(np.arange(self.nepochs), self.err)
        axes.set_xlabel('epochs')
        axes.set_ylabel('error')
        plt.show()
        fig.savefig('../images/training_plot_output_mlp_exp{exp}.png'.format(exp=self.exp))
        self.plot(yr)
    
dt = 0.0010
N = 500
nepochs = 30000
num_osc = 20
num_h = 200
num_out = 8
lr = 0.0010
exp = 4
train = Train(dt, N, nepochs, num_osc, num_h, num_out, exp, 'random', lr)
train()
