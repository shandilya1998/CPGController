import numpy as np
from gait_generation import gait_generator as gg
from frequency_analysis import frequency_estimator
from tqdm import tqdm
import scipy.fft as fft 
import scipy.signal as signal 

version = 4 

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
        self.signal =  gg.get_signal(self.dt, Tsw, Tst, self.N, theta, version)
        self.preprocess()

    def get_ff(self, signal, ff_type = 'fft'):
        if ff_type == 'fft': 
            return frequency_estimator.freq_from_fft(signal, 1/self.dt)
        elif ff_type == 'autocorr':
            return frequency_estimator.freq_from_autocorr(signal, 1/self.dt)    
    
    def set_num_osc(self, num):
        self.num_osc = num 
  
    def get_output(self, Tsw, Tst, theta):
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
            self.Y.append(datapoint.get_output(Tsw[i], Tst[i], theta[i]))
            self.X.append(datapoint)
        return self.X, self.Y
