import numpy as np

class OscLayer(object):
    def __init__(self, num_osc, N, dt):
        self.N = N
        self.dt = dt
        self.num_osc = num_osc

    def get_signal(self, freq):
        Z = np.empty((self.num_osc, self.N), dtype = np.complex128)
        r = np.empty((self.num_osc, self.N), dtype = np.complex128)
        phi = np.empty((self.num_osc, self.N))
        for i in range(self.N-1):
            r[:, i+1] = r[:, i] + ((1-np.abs(r[: ,i])**2)*r[:, i]+1j*freq*r[:, i])*self.dt 
            phi[:, i+1] = phi[:, i] + freq*self.dt
            Z[:, i+1] = r[:, i+1]*np.exp(1j*2*np.pi*phi[:, i+1])
        return Z 
