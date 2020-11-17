import numpy as np
from COM import COM
from support_plane import SupportPlane
from dynamics import QuadrupedDynamics, KneeFourBarDynamics

class AxisTransformation:
    def __init__(self, L = None, M = None, m = None, Tb = None):
        self.support_plane_obj = SupportPlane(Tb)
        self.L = L
        self.M = M
        self.m = m
        self.Tb = Tb

    def setup(self, t, A, B, AL, AF, BF, BL):
        self.support_plane_obj.setup(A, B, AL, AF, BF, BL)
        self.support_plane = self.support_plane_obj.get_support_plane(t).T
        
    def com_s(self, theta):
        return np.matmul(self.support_plane, COM(L = self.L, M = self.M, m = self.m)(theta))

    def transform(self, vec):
        return np.matmul(self.support_plane, vec)

    def inverse_transform(self, vec):
        return np.matmul(np.inverse(self.support_planee), vec)

class ZMP:
    def __init__(self, M, g, T, B, fr, mu, m, g, Tb, m1, m2, m3, L0, L2, L2, L3, I1, I2, I3, L = None):
        self.dynamic_model = QuadrupedDynamics(M, g, T, B, fr, mu, m1, m2, m3, L0, L2, L2, L3, I1, I2, I3)
        self.transformation = AxisTransformation(L = L, M = M, m = m, Tb = Tb)
        self.G = np.array([0, 0, g])
        self.M = M  

    def setup(self, t, A, B, AL, AF, BF, BL):
        self.transformation.setup(t, A, B, AL, AF, BF, BL)
  
    def solve_dynamics(self, gamma, theta, t11, phi1, o11z, a11z,  t21, phi2, o21z, a21z,  t31, phi3, o31z, a31z,  t41, phi4, o41z, a41z):
        self.dynamic_model.setup(gamma, theta, t11, phi1, o11z, a11z,  t21, phi2, o21z, a21z,  t31, phi3, o31z, a31z,  t41, phi4, o41z, a41z)
        A = self.dynamic_model.A()
        B = self.dynamic_model.B()
        return np.matmul(np.linalg.inv(A), B)
        
    def zmp_s(self, F_c, N_c, theta):
        """
            F_c : inertial forces acting at the center of mass
            N_c : inertial moments acting at on the quadruped
        """
        F_s = self.transformation.transform(F_c)
        N_s = self.transformation.transform(N_c)
        com_s = self.transformation.com_s(theta)
        G_s = self.transformation.transform(self.g)*self.M
        zmp_s = np.zeros(3)
        zmp_s[0] = 0
        zmp_s[1] = com_s[1] - (com_s[0]*(F_s[1] + G_s[1]) + N_s[2])/(F_s[0] + G_s[0])
        zmp_s[2] = com_s[2] - (com_s[0]*(F_s[2] + G_s[1]) - N_s[1])/(F_s[0] + G_s[0]) 
        return zmp_s

class ModZMP:
    def __init__(self, M, g, T, B, fr, mu, m, g, Tb, m1, m2, m3, L0, L2, L2, L3, I1, I2, I3, L = None):
        self.zmp = ZMP(M, g, T, B, fr, mu, m, g, Tb, m1, m2, m3, L0, L2, L2, L3, I1, I2, I3, L)
        
    def setup(self, t, A, B, AL, AF, BF, BL):
        self.zmp.setup(t, A, B, AL, AF, BF, BL)

    def mod_zmp(self, eta, v_d, v_r, gamma, theta, t11, phi1, o11z, a11z,  t21, phi2, o21z, a21z,  t31, phi3, o31z, a31z,  t41, phi4, o41z, a41z):
        zmp = self.zmp.zmp_s(gamma, theta, t11, phi1, o11z, a11z,  t21, phi2, o21z, a21z,  t31, phi3, o31z, a31z,  t41, phi4, o41z, a41z)
        return zmp+eta*(self.zmp.transformation.transform(v_r)-self.zmp.transformation.transform(v_d))
