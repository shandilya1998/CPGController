import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import scipy.signal as sgnl
import scipy.fft as fft
import gait_generator as gg
from tqdm import tqdm
import frequency_estimator as fe

dt = 0.001
N = 10000
n = 100

class FundamentalFrequencyVsSpeedAnalysis(object):
    def __init__(self, dt, N, n):
        self.n = n
        self.dt = dt
        self.N = N
        #setting default swing and stance timings
        self.Tsw = 20
        self.Tst = 60
        self.theta = 15
        self.set_speed()
        self.set_signal()        

    def set_speed(self):
        self.v = 2*self.theta*0.06/((self.Tsw+self.Tst)*self.dt)


    def set_signal(self):
        self.signal = gg.get_signal(self.dt,  
                                    self.Tsw, 
                                    self.Tst,
                                    self.N,
                                    self.theta) 
        
    def set_Tsw(self, tsw):
        self.Tsw = tsw
        self.set_signal()
        self.set_speed()

    def set_Tst(self, tst):
        self.Tst = tst
        self.set_signal()
        self.set_speed()

    def set_theta(self, theta):
        self.theta = theta
        self.set_signal()
        self.set_speed()
    
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
    
    def _find_gcd(self, x, y): 
        while(y): 
            x, y = y, x % y 
        return x     
    
    def find_gcd(self, lst):
        gcd = self._find_gcd(lst[0], lst[1])
        for i in range(2, len(lst)):
            gcd = self._find_gcd(gcd, lst[i])
        return gcd

    def __call__(self):        
        cols = [ 
                'speed', 
                'Tsw',
                'Tst',
                'theta',
                'Fundamental Frequency Hip 1',
                'Fundamental Frequency Knee 1',
                'Fundamental Frequency Hip 2',
                'Fundamental Frequency Knee 2', 
                'Fundamental Frequency Hip 3',
                'Fundamental Frequency Knee 3', 
                'Fundamental Frequency Hip 4',
                'Fundamental Frequency Knee 4' 
                ]
        df = pd.DataFrame(columns = cols)
        print(df.columns)
        inc = 2
        count = 0
        for i in tqdm(range(self.n)):
            for j in range(self.n):
                lst = [self.v, self.Tsw, self.Tst, self.theta]
                for j in range(8):
                    ff = fe.freq_from_autocorr(self.signal[:, j+1], 1/self.dt) 
                    lst.append(ff)
                    count+=1
                df.loc[count] = lst
                self.set_Tst(self.Tst+inc)
            self.set_Tsw(self.Tsw+inc)
            self.set_Tst(3*self.Tsw) 
        df.to_csv('fundamental_freq_analysis.csv', index = False)
        for i in tqdm(range(self.n)):
            lst = [self.v, self.Tsw, self.Tst, self.theta]
            for j in range(8):
                ff = fe.freq_from_autocorr(self.signal[:, j+1], 1/self.dt) 
                lst.append(ff)
                count+=1
            df.loc[count] = lst 
            self.set_theta(self.theta+inc)
        df.to_csv('fundamental_freq_vs_theta_analysis.csv', index = False) 
        
ob = FundamentalFrequencyVsSpeedAnalysis(dt, N, n)
ob()        
