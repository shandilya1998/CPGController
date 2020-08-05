import numpy as np

class OscLayer(object):
    def __init__(self, num_osc, N, dt):
        self.N = N
        self.dt = dt
        #print(dt)
        self.num_osc = num_osc

    def get_signal(self, freq):
        Z = np.zeros((self.num_osc, self.N), dtype = np.complex128)
        r1 = np.random.random((self.num_osc, self.N))
        r2 = np.random.random((self.num_osc, self.N)) 
        Z = r1 + 1j*r2 
        #print(Z.shape)
        for i in range(self.N-1):
            Z[:, i+1] = Z[:, i] + ((1-np.abs(Z[: ,i])**2)*Z[:, i]+1j*freq*Z[:, i])*self.dt 
        #print(Z)
        return Z 
