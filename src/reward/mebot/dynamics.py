import numpy as np
from reward.kinematics import KneeFourBarKinematics

class KneeFourBarDynamics:
    def __init__(self, m1, m2, m3, L0, L1, L2, L3, g):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.g = g
        self.kinematics = KneeFourBarKinematics(L0, L1, L2, L3)
        self.theta = np.zeros(4)
        self.r = np.zeros(4)
        self.omega = np.zeros(4)
        self.alpha = np.zeros(4)
        self.a = np.zeros(4)
        self.gamma = 0
        
       
    def build(self, t1, phi, o1z, a1z, gamma, I1, I2, I3):
        self.I1 = I1
        self.I2 = I2
        self.I3 = I3
        self.theta = self.kinematics.theta(t1, phi)
        self.r = self.kinematics.r(self.theta)
        self.omega = self.kinematics.omega(r, o1z)
        self.alpha = self.kinematics.alpha(r, omega, a1z)
        self.a = self.kinematics.a(r, omega, alpha)
        self.gamma = gamma 
    
 
    def A(self):
        """
            The following are the indices of all the variables involved in
            the dynamics of each leg:
                F01x : 0    F01y : 1    F21x : 2    F21y : 3
                F12x : 4    F12y : 5    F32x : 6    F32y : 7
                F23x : 8    F23y : 9    F03x: 10    F03y: 11
                T : 12      N : 13
        """
        A = np.zeros((13, 14))

        A[0][0] = 1
        A[0][2] = 1
        
        A[1][1] = 1
        A[1][3] = 1
        
        A[2][12] = 1
        A[2][0] = self.L1 * np.sin(self.theta[1] - self.theta[0] + np.pi) / 2
        A[2][2] = - self.L1 * np.sin(self.theta[1] - self.theta[0] + np.pi) / 2
        A[2][1] = - self.L1 * np.cos(self.theta[1] - self.theta[0] + np.pi) / 2
        A[2][3] = self.L1 * np.cos(self.theta[1] - self.theta[0] + np.pi) / 2

        A[3][4] = 1
        A[3][6] = 1
        A[3][13] = gamma * np.cos(self.theta[2] - self.theta[0] + np.pi)
        
        A[4][5] = 1
        A[4][7] = 1
        A[4][13] = gamma * np.sin(self.theta[2] - self.theta[0] + np.pi)
    
        A[5][4] = self.L2 * np.sin(self.theta[2] - self.theta[0] + np.pi) / 2
        A[5][6] = - self.L2 * np.sin(self.theta[2] - self.theta[0] + np.pi) / 2
        A[5][5] = - self.L2 * np.cos(self.theta[2] - self.theta[0] + np.pi) / 2
        A[5][7] = self.L2 * np.cos(self.theta[2] - self.theta[0] + np.pi) / 2
        
        A[6][10] = 1
        A[6][8] = 1

        A[7][11] = 1
        A[7][9] = 1

        A[8][10] = self.L3 * np.sin(self.theta[3] - self.theta[0] + np.pi) / 2
        A[8][8] = -self.L3 * np.sin(self.theta[3] - self.theta[0] + np.pi) / 2
        A[8][11] = -self.L3 * np.cos(self.theta[3] - self.theta[0] + np.pi) / 2
        A[8][9] = self.L3 * np.cos(self.theta[3] - self.theta[0] + np.pi) / 2

        A[9][4] = 1
        A[9][2] = 1

        A[10][5] = 1
        A[10][3] = 1

        A[11][8] = 1
        A[11][6] = 1

        A[12][9] = 1
        A[12][7] = 1

        return A
        
    def B(self):
        B = np.zeros(13)

        B[0] = self.m1 * self.a[1][0] + self.m1 * self.g * \
                np.cos(self.theta[0] - 3 * np.pi / 2)
        B[1] = self.m1 * self.a[1][1] + self.m1 * self.g * \
                np.sin(self.theta[0] - 3 * np.pi / 2)
        B[2] = self.I1 * self.alpha[1][2]
        B[3] = self.m2 * self.a[2][0] + self.m1 * self.g * \
                np.cos(self.theta[0] - 3 * np.pi / 2)
        B[4] = self.m2 * self.a[2][1] + self.m1 * self.g * \
                np.sin(self.theta[0] - 3 * np.pi / 2)
        B[5] = self.I2 * self.alpha[2][2]
        B[6] = self.m3 * self.a[3][0] + self.m1 * self.g * \
                np.cos(self.theta[0] - 3 * np.pi / 2)
        B[7] = self.m3 * self.a[3][1] + self.m1 * self.g * \
                np.sin(self.theta[0] - 3 * np.pi / 2)
        B[8] = self.I3 * self.alpha[3][2]

        return B


class QuadrupedDynamics:
    def __init__(self, M, g, T, Bt, fr, mu, m1, m2, m3, L0, L1, L2, L3):
        self.M = M
        self.g = g
        self.mu = mu
        self.T = T
        self.Bt = Bt
        self.fr = fr
        self.gamma = np.zeros(4)
        self.theta = np.zeros(4)
        self.knee1 = KneeFourBarDynamics(m1, m2, m3, L0, L1, L2, L3, g)
        self.knee2 = KneeFourBarDynamics(m1, m2, m3, L0, L1, L2, L3, g)
        self.knee3 = KneeFourBarDynamics(m1, m2, m3, L0, L1, L2, L3, g)
        self.knee4 = KneeFourBarDynamics(m1, m2, m3, L0, L1, L2, L3, g)  

    def build(
            self, gamma, theta, 
            t11, phi1, o11z, a11z,  
            t21, phi2, o21z, a21z,  
            t31, phi3, o31z, a31z,  
            t41, phi4, o41z, a41z,
            I1, I2, I3
        ):
        self.I1 = I1
        self.I2 = I2
        self.I3 = I3
        self.gamma = gamma
        self.theta = theta
        self.knee1.build(t11, phi1, o11z, a11z, gamma[0], \
                        I1[0], I2[0], I3[0])
        self.knee2.build(t21, phi2, o21z, a21z, gamma[1], \
                        I1[1], I2[1], I3[1])
        self.knee3.build(t31, phi3, o31z, a31z, gamma[2], \
                        I1[2], I2[2], I3[2])
        self.knee4.build(t41, phi4, o41z, a41z, gamma[3], \
                        I1[3], I2[3], I3[3])

    def A(self):
        """
            The following are the indices of the variables involved in the 
            dynamics of the quadruped:
                [0, 14) : knee 1 variables   [14, 28) : knee 2 variables
                [28, 42) : knee 3 variables  [42, 56) : knee 4 variables
                56 : Ffr1   57 : Ffr2   58 : Ffr3   59 : Ffr3
                60 : Fx   61 : Fy   62 : Nx    63 : Ny   
                64 : Nz   62 : Th1   63 : Th2   64 : Th3   65 : Th4
        """
        A = np.zeros((69, 69))
        A1 = self.knee1.A()
        A2 = self.knee2.A()
        A3 = self.knee3.A()
        A4 = self.knee4.A()
        
        A[:13][:14] = A1
        A[13:26][14:28] = A2
        A[26:39][28:42] = A3
        A[39:52][42:56] = A4
        
        A[52][56] = np.cos(self.theta[0])
        A[52][57] = np.cos(np.pi - self.theta[1])
        A[52][58] = np.cos(3*np.pi/2 - self.theta[2])
        A[52][59] = np.cos(self.theta[3] - 3*np.pi/2) 
        A[52][61] = -1

        A[53][56] = np.sin(self.theta[0])
        A[53][57] = -np.sin(np.pi - self.theta[1])
        A[53][58] = np.sin(3*np.pi/2 - self.theta[2])
        A[53][59] = -np.sin(self.theta[3] - 3*np.pi/2)
        A[53][60] = -1

        A[54][13] = self.gamma[0]
        A[54][27] = self.gamma[1]
        A[54][41] = self.gamma[2]
        A[54][55] = self.gamma[3]

        A[55][56] = 1
        A[55][13] = -self.mu

        A[56][57] = 1
        A[56][27] = -self.mu
        
        A[57][58] = -1
        A[57][41] = -self.mu

        A[58][59] = -1
        A[58][55] = -self.mu

        lst = []
        lst_ = []
        for i in range(4):
            if self.gamma[i] == 1:
                lst.append(i)
            else:
                lst_.append(i)
        
        if len(lst) == 3:
            A[59][14*lst[0] - 1] = 1
            A[59][14*lst[1] - 1] = -1

            A[60][14*lst[0] - 1] = 1
            A[60][14*lst[2] - 1] = -1
        
            A[61][14*lst_[0] - 1] = 1

        elif len(lst) == 2:
            A[59][14*lst[0] - 1] = 1 
            A[59][14*lst[1] - 1] = -1 

            A[60][14*lst_[0] - 1] = 1
            
            A[61][14*lst_[1] - 1] = 1

        A[62][56] = self.gamma[0] * \
                    (
                        np.cos(self.theta[0] - np.pi/2) * \
                        (self.Bt + self.T*np.cos(self.theta[0])) - \
                        np.sin(self.theta[0] - np.pi/2)* \
                        (self.Bt + self.T*np.sin(self.theta[0]))
                    )
        A[62][57] = self.gamma[1] * \
                    (
                        np.cos(self.theta[1] + np.pi/2) * \
                        (-self.Bt + self.T*np.cos(self.theta[1])) - \
                        np.sin(self.theta[1] - np.pi/2) * \
                        (self.Bt + self.T*np.sin(self.theta[1]))
                    )
        A[62][58] = self.gamma[2] * \
                    (
                        np.cos(self.theta[2] + np.pi/2) * \
                        (-self.Bt + self.T*np.cos(self.theta[2])) - \
                        np.sin(self.theta[2] - np.pi/2) * \
                        (-self.Bt + self.T*np.sin(self.theta[2]))
                    )
        A[62][59] = self.gamma[3] * \
                    (
                        np.cos(self.theta[3] - np.pi/2) * \
                        (self.Bt + self.T*np.cos(self.theta[3])) - \
                        np.sin(self.theta[3] - np.pi/2) * \
                        (-self.Bt + self.T*np.sin(self.theta[3]))
                    )
        A[62][64] = -1

        A[63][13] = self.gamma[0]*(self.Bt + self.T*np.cos(self.theta[0]))
        A[63][27] = self.gamma[1]*(-self.Bt + self.T*np.cos(self.theta[1]))
        A[63][41] = self.gamma[2]*(-self.Bt + self.T*np.cos(self.theta[2]))
        A[63][55] = self.gamma[3]*(self.Bt + self.T*np.cos(self.theta[3]))
        A[63][63] = -1

        A[64][13] = self.gamma[0]*(self.Bt + self.T*np.sin(self.theta[0]))
        A[64][27] = self.gamma[1]*(self.Bt + self.T*np.sin(self.theta[1]))
        A[64][41] = self.gamma[2]*(-self.Bt + self.T*np.sin(self.theta[2]))
        A[64][55] = self.gamma[3]*(-self.Bt + self.T*np.sin(self.theta[3]))
        A[64][62] = -1

        A[65][65] = 1
        A[65][56] = self.gamma[0]*self.T

        A[66][66] = 1
        A[66][57] = self.gamma[1]*self.T

        A[67][67] = 1
        A[67][58] = self.gamma[2]*self.T

        A[68][68] = 1
        A[68][59] = self.gamma[3]*self.T

        return A

    def B(self):
        B = np.zeros(69)
        B1 = self.knee1.B()
        B2 = self.knee2.B()
        B3 = self.knee3.B()
        B4 = self.knee4.B()
        
        B[:13] = B1
        B[13:26] = B2
        B[26:39] = B3
        B[39:52] = B4

        B[54] = self.M*self.g

        B[65] = (1 - self.gamma[0])*self.fr
        B[66] = (1 - self.gamma[1])*self.fr
        B[67] = (1 - self.gamma[2])*self.fr
        B[68] = (1 - self.gamma[3])*self.fr

        return B 
