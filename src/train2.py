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

class DataPoint(object):
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
        Z = np.empty((self.num_osc, self.N), dtype = np.complex128)
        self.set_signal(Tsw, Tst, theta)
        """ 
            Assuming a straight line motion with a constant speed
        """
        ff = self.get_ff(self.signal[:, 1], 'autocorr')
        freq = np.empty((self.num_osc,))
        for i in range(self.num_osc):
            freq[i] = 2*np.pi*ff*(i+1)
        return self.osc.get_signal(freq)

class DataLoader(object):
    def __init__(self, dt, num_osc):
        self.dt = dt
        self.num_osc = num_osc
        self.Y = []
        self.X = []

    def __call__(self, Tsw, Tst, theta, N, num):
        for i in range(num):
            datapoint = DataPoint(self.dt, N[i], self.num_osc)
            self.Y.append(datapoint.get_input(Tsw[i], Tst[i], theta[i]))
            self.X.append(datapoint)
        return self.X, self.Y

class Train(object):
    def __init__(self, 
                 dt, 
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
        self.nepochs = nepochs
        self.data = DataLoader(self.dt, self.num_osc)
        self.out_mlp = OutputMLP()
        self.lr = lr
        self.err = np.zeros((self.nepochs,))
        self.exp = exp

    def _plot(self, axis, x, y, t):
        axis.plot(
            t, 
            x,
            'r',
            label = 'ideal gait')
        axis.plot(
            t, 
            y, 
            'b',
            label = 'generated gait')
        axis.set_xlabel('time')
        axis.set_ylabel('joint activation')
        axis.legend()    

    def plot(self, yr, signal):
        fig, axes = plt.subplots(self.num_out, 1, figsize = (5, 5*self.num_out))
        if self.num_out!=1:
            for i in range(self.num_out):
                self._plot(axes[i], signal[:, i+1].T, yr[i], signal[:, 0].T)
        else:
            self._plot(axes, signal[:, 1].T, yr[0], signal[:, 0].T)
        fig.savefig('../images/pred_vs_ideal_exp{exp}.png'.format(exp=self.exp))
        plt.show()

    def __call__(self, 
                 Tsw,
                 Tst,
                 theta, 
                 N,
                 num):
        self.out_mlp.build(self.num_osc, 
                           self.num_h, 
                           self.num_out,
                           0.5, 
                           self.init)
        """
            Assuming a straight line motion with a constant speed
        """
        X, Y = self.data(Tsw, Tst, theta, N, num)
        for i in tqdm(range(self.nepochs)):
            for j in range(num-1):
                yr = self.out_mlp(Y[j], self.out_mlp.sigmoidf)
                err = np.sum((X[j].signal[:, 1:self.num_out+1].T-yr)**2)
                #print(err)
                self.err[i] = err
                """
                    Back propagation
                """
                self.out_mlp.backprop_sigmoid(yr, Y[j], X[j], self.lr)
                if i>50 and math.isclose(
                    self.err[i],
                    self.err[i-50],
                    rel_tol = 1e-5,
                    abs_tol=1e-5,
                ):
                    self.lr = self.lr*0.9
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

        yr = self.out_mlp(Y[num-1], self.out_mlp.sigmoidf)
        pkl = open('weights/exp{exp}/w2_out_mlp.pickle'.format(exp=self.exp), 'wb')
        pickle.dump(self.out_mlp.W2, pkl)
        pkl.close()
        pkl = open('weights/exp{exp}/w1_out_mlp.pickle'.format(exp=self.exp), 'wb')
        pickle.dump(self.out_mlp.W1, pkl)
        pkl.close()
        fig, axes = plt.subplots(1, 1, figsize = (5, 5))
        axes.plot(np.arange(self.nepochs), self.err)
        axes.set_xlabel('epochs')
        axes.set_ylabel('error')
        plt.show()
        fig.savefig('../images/training_plot_output_mlp_exp{exp}.png'.format(exp=self.exp))
        self.plot(yr, X[num-1].signal)
   
 
dt = 0.001
nepochs = 3000
num_osc = 20
num_h = 50
num_out = 8
lr = 1e-3
exp = 5
num = 6
Tst = np.array([60, 80, 100, 120, 140, 75])
Tsw = np.array([20, 26, 33, 40, 46, 25 ])
theta = np.array([15, 15, 15, 15, 15, 15])
N = np.array([400, 530, 665, 800, 930, 500])
train = Train(dt, nepochs, num_osc, num_h, num_out, exp, 'random', lr)
train(Tsw, Tst, theta, N, num)
