import numpy as np
from mod_zmp import ModZMP

class FitnessFunction:
    def __init__(self, M, g, T, B, fr, mu, m, g, Tb, m1, m2, m3, L0, L2, L2, L3, I1, I2, I3, L = None):
        self.mod_zmp = ModZMP(M, g, T, B, fr, mu, m, g, Tb, m1, m2, m3, L0, L2, L2, L3, I1, I2, I3, L)
    
    def setup(self, t, A, B, AL, AF, BF, BL):
        self.A = A
        self.B = B
        self.mod_zmp.setup(t, A, B, AL, AF, BF, BL)

    def loss(self, eta, v_d, v_r, gamma, theta, t11, phi1, o11z, a11z,  t21, phi2, o21z, a21z,  t31, phi3, o31z, a31z,  t41, phi4, o41z, a41z):
        try:
            zmp = self.mod_zmp.mod_zmp(eta, v_d, v_r, gamma, theta, t11, phi1, o11z, a11z, t21, phi2, o21z, a21z, t31, phi3, o31z, a31z, t41, phi4, o41z, a41z)
            AC = self.mod_zmp.zmp.transformation.inverse_transform(zmp) - self.A
            return np.abs(np.cross(AC, self.B - self.A))/np.abs(self.B-self.A)
        except AttributeError:
            raise AttributeError('loss called before setup')
    
