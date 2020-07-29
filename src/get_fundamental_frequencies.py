import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import scipy.fft as fft
import gait_generator as gg

dt = 0.001
N = 1000000

class FundamentalFrequencyVsSpeedAnalysis(object):
    def __init__(self, dt, N):
        self.dt = dt
        self.N = N
        #setting default swing and stance timings
        self.Tsw = 50
        self.Tst = 150
        self.theta = 15
        self.set_speed()
        self.set_signal()        

    def set_speed(self):
        self.v = 2*self.theta*0.06/(self.Tsw+self.Tst)


    def set_signal(self):
        self.signal = gg.get_signal(self.dt,  
                                    self.Tsw, 
                                    self.Tst,
                                    self.N,
                                    self.theta) 
        
    def set_Tsw(self, tsw):
        self.Tsw = tsw
        self.set_signal()

    def set_Tst(self, tst):
        self.Tst = tst
        self.set_signal()

    def set_theta(self, theta):
        self.theta = theta
        self.set_signal()
    
    def plot_magnitude_spectrum(self, signal):
        FT = fft.fft(signal)
        t = self.signal[:, 0]
        fig, ax = plt.subplots(2,1, figsize =(5, 10))
        ax[0].plot(t[:int(self.N*self.dt)], signal[:int(self.N*self.dt)])
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Signal amplitude')       
        freqs = fft.fftfreq(len(signal), 1/self.dt)
        ax[1].plot(freqs, np.abs(FT))
        ax[1].set_xlabel('Frequency')
        ax[1].set_ylabel('Magnitude')
        fig.savefig('../images/magnitude_spectrum_Tst_{Tst}_Tsw_{Tsw}_theta_{theta}'.format(Tst = self.Tst, Tsw = self.Tsw, theta = self.theta))
    
    def __call__(self):        
        print(self.signal.shape)
        self.plot_magnitude_spectrum(self.signal[:,1])
        plt.show()    

ob = FundamentalFrequencyVsSpeedAnalysis(dt, N)
ob()        
