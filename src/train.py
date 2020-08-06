import numpy as np
from gait_generation import gait_generator as gg
from frequency_analysis import frequency_estimator
from tqdm import tqdm
from layers.OutputMLP import OutputMLP
from layers.OscLayer import OscLayer
import pickle
import matplotlib.pyplot as plt

class DataLoader(object):
    def __init__(self, 
                 dt, 
                 N,
                 num_osc):
        self.dt = dt
        self.N = N
        self.num_osc = num_osc
        self.osc = OscLayer(self.num_osc, self.N, self.dt)

    def set_signal(self, Tsw, Tst, theta):
        self.signal =  gg.get_signal(self.dt, Tsw, Tst, self.N, theta)

    def get_ff(self, signal): 
        return frequency_estimator.freq_from_autocorr(signal, 1/self.dt)

    def set_num_osc(self, num):
        self.num_osc = num
  
    def get_input(self, Tsw, Tst, theta):
        Z = np.empty((self.num_osc, N), dtype = np.complex128)
        self.set_signal(Tsw, Tst, theta)
        """
            Assuming a straight line motion with a constant speed
        """
        ff = self.get_ff(self.signal[:, 1])
        freq = np.empty((self.num_osc,))
        for i in range(self.num_osc):
            freq[i] = 2*np.pi*ff*(i+1)
        return self.osc.get_signal(freq)
                
class Train(object):
    def __init__(self, 
                 dt, 
                 N, 
                 nepochs,
                 num_osc, 
                 num_h, 
                 num_out, 
                 init = 'random',
                 lr = 1e-3):
        self.num_osc = num_osc
        self.num_h = num_h
        self.num_out = num_out
        self.init = init
        self.dt = dt
        self.N = N
        self.nepochs = nepochs
        self.out_mlp = OutputMLP()
        self.data = DataLoader(self.dt, self.N, self.num_osc)
        self.lr = lr
        self.err = np.zeros((self.nepochs,))

    def _plot(self, axis, x, y):
        axis.plot(
            self.data.signal[:, 0].T, 
            x,
            'r',
            label = 'ideal gait')
        axis.plot(
            self.data.signal[:, 0].T, 
            y, 
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
        fig.savefig('../images/pred_vs_ideal_exp2.png')
        plt.show()

    def __call__(self):
        self.out_mlp.build(self.num_osc, 
                           self.num_h, 
                           self.num_out, self.init)
        """
            Assuming a straight line motion with a constant speed
        """
        Tst = 60
        Tsw = 10
        theta = 15
        Z = self.data.get_input(Tsw, Tst, theta)
        for i in tqdm(range(self.nepochs)):
            yr = self.out_mlp(Z)
            err = np.sum((self.data.signal[:, 1:self.num_out+1].T-yr)**2)
            #print(err)
            self.err[i] = err            
            """
                Back propagation
            """
            dW2r = -1*np.matmul(
                        (self.data.signal[:, 1:self.num_out+1].T-yr)*(1+yr)*(1-yr)/2, 
                        self.out_mlp.xhr.T)
            dW2i = np.matmul(
                        (self.data.signal[:, 1:self.num_out+1].T-yr)*(1+yr)*(1-yr)/2, 
                        self.out_mlp.xhi.T)
            dW1r = -1*np.matmul(
                        np.matmul(
                            self.out_mlp.W2.real.T, 
                            (self.data.signal[:, 1:self.num_out+1].T-yr)*(1+yr)*(1-yr)/2)*(1+self.out_mlp.xhr)*(1-self.out_mlp.xhr)/2, 
                        Z.real.T) + \
                    np.matmul(
                        np.matmul(
                            self.out_mlp.W2.imag.T, 
                            (self.data.signal[:, 1:self.num_out+1].T-yr)*(1+yr)*(1-yr)/2)*(1+self.out_mlp.xhi)*(1-self.out_mlp.xhi)/2, 
                        Z.imag.T)
            dW1i = np.matmul(
                        np.matmul(self.out_mlp.W2.real.T, 
                                 (self.data.signal[:, 1:self.num_out+1].T-yr)*(1+yr)*(1-yr)/2)*(1+self.out_mlp.xhr)*(1-self.out_mlp.xhr)/2, 
                        Z.imag.T) + \
                    np.matmul(
                        np.matmul(
                            self.out_mlp.W2.imag.T, 
                            (self.data.signal[:, 1:self.num_out+1].T-yr)*(1+yr)*(1-yr)/2)*(1+self.out_mlp.xhi)*(1-self.out_mlp.xhi)/2, 
                        Z.real.T) 
            W2r = self.out_mlp.W2.real - self.lr*dW2r 
            W2i = self.out_mlp.W2.imag - self.lr*dW2i
            self.out_mlp.set_W2(W2r + 1j*W2i)
            W1r = self.out_mlp.W1.real - self.lr*dW1r
            W1i = self.out_mlp.W1.imag - self.lr*dW1i
            self.out_mlp.set_W1(W1r+1j*W1i)
            if i>20 and self.err[i]==self.err[i-10]:
                break
                
        yr = self.out_mlp(Z)
        pkl = open('weights/exp2/w2_out_mlp.pickle', 'wb')
        pickle.dump(self.out_mlp.W2, pkl)
        pkl.close()
        pkl = open('weights/exp2/w1_out_mlp.pickle', 'wb')
        pickle.dump(self.out_mlp.W1, pkl)
        pkl.close()
        fig, axes = plt.subplots(1, 1, figsize = (5, 5))
        axes.plot(np.arange(self.nepochs), self.err)
        axes.set_xlabel('epochs')
        axes.set_ylabel('error')
        plt.show()
        fig.savefig('../images/training_plot_output_mlp_exp2.png')
        self.plot(yr)
    
dt = 0.001
N = 500
nepochs = 3000
num_osc = 100
num_h = 200
num_out = 1 
lr = 0.00001
train = Train(dt, N, nepochs, num_osc, num_h, num_out, 'random', lr)
train()
