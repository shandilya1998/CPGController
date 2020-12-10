import numpy as np
import tensorflow as tf

class OscLayer(object):
    def __init__(self, num_osc, N, dt):
        self.N = N
        self.dt = dt
        #print(dt)
        self.num_osc = num_osc

    def get_signal(self, freq):
        Z = np.zeros((self.num_osc, self.N), dtype = np.complex128)
        #Zi = np.zeros((self.num_osc, self.N))
        r = np.ones((self.num_osc, self.N))
        phi = np.zeros((self.num_osc, self.N))
        Z[:, 0] = r[:, 0]*np.exp(1j*phi[:, 0])
        #Zi[:, 0] = r[:, 0]*np.exp(1j*phi[:, 0]).imag
        for i in range(self.N-1):
            r[:, i+1] = r[:, i] + (1-r[:, i]**2)*r[:, i]*self.dt
            #print(r[:, i+1])
            #print("\n")
            phi[:, i+1] = phi[:, i] + freq*self.dt
            #print(phi[:, i+1])
            #print("\t")
            Z[:, i+1] = r[:, i+1]*np.exp(1j*phi[:, i+1])
            #Zi[:, i+1] = r[:, i+1]*np.exp(1j*phi[:, i+1]).imag
        return Z#r* 1j*Zi

