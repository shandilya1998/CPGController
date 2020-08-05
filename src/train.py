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
            freq[i] = ff*(i+1)
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
                 lr = 0.001):
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

    def __call__(self):
        self.out_mlp.build(self.num_osc, 
                           self.num_h, 
                           self.num_out, self.init)
        """
            Assuming a straight line motion with a constant speed
        """
        Tst = 60
        Tsw = 20
        theta = 15
        Z = self.data.get_input(Tst, Tsw, theta)
        for i in tqdm(range(self.nepochs)):
            yr = self.out_mlp(Z)
            print(yr)
            err = np.sum(yr-self.data.signal[:, 1:].T)**2
            self.err[i] = err            
            """
                Back propagation
            """
            dW2r = -1*np.matmul((self.data.signal[:, 1:].T-yr)*yr*(1-yr), self.out_mlp.xhr.T)
            dW2i = np.matmul((self.data.signal[:, 1:].T-yr)*yr*(1-yr), self.out_mlp.xhi.T)
            dW1r = -1*np.matmul(np.matmul(self.out_mlp.W2.real.T, 
                                          (self.data.signal[:, 1:].T-yr)*yr*(1-yr)), 
                                Z.real.T) + np.matmul(np.matmul(self.out_mlp.W2.imag.T, 
                                                              (self.data.signal[:, 1:].T-yr)*yr*(1-yr)), 
                                                      Z.imag.T)
            dW1i = np.matmul(np.matmul(self.out_mlp.W2.real.T, 
                                       (self.data.signal[:, 1:].T-yr)*yr*(1-yr)), 
                             Z.imag.T) + np.matmul(np.matmul(self.out_mlp.W2.imag.T, 
                                                             (self.data.signal[:, 1:].T-yr)*yr*(1-yr)), 
                                                   Z.real.T) 
            W2r = self.out_mlp.W2.real - self.lr*dW2r
            W2i = self.out_mlp.W2.imag - self.lr*dW2i
            self.out_mlp.set_W2(W2r + 1j*W2i)
            W1r = self.out_mlp.W1.real - self.lr*dW1r
            W1i = self.out_mlp.W1.imag - self.lr*dW1i
            self.out_mlp.set_W1(W1r+1j*W1i)
        
        pkl = open('w2_out_mlp.pickle', 'wb')
        pickle.dump(self.out_mlp.W2)
        pkl.close()
        pkl = open('w1_out_mlp.pickle', 'wb')
        pickle.dump(self.out_mlp.W1)
        pkl.close()
        fig, axes = plt.subplots(1,1 (5, 5))
        axes.plot(np.arange(self.nepochs), self.err)
        axes.set_xlabel('epochs')
        axes.set_ylabel('error')
        plt.show()
        fig.savefig('../images/training_plot_output_mlp_exp1.png')

dt = 0.001
N = 100000
nepochs = 3000
num_osc = 10
num_h = 20
num_out = 8
train = Train(dt, N, nepochs, num_osc, num_h, num_out)
train()
