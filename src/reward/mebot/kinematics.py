import numpy as np

class LegKinematics:
    def __init__(self, thigh, leg):
        self.t = thigh
        self.leg = leg
        self.built = False

    def build(self):
        self.built = True

        

class KneeFourBarKinematics:

    def __init__(self, L0, L1, L2, L3):
        self.L0 = L0
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

        self.built = False

    def build(self):
        self.built = True

    def theta(self, t1, phi):
        theta = np.zeros(4)
        theta[0] = phi
        theta[1] = t1
        l = np.sqrt(self.L0**2+self.L1**2-2*self.L0*self.L1*np.cos(t1-phi+np.pi))
        b1 = np.arcsin(self.L1*np.sin(t1-phi+np.pi)/l)
        b2 = np.arccos((self.L2**2+l**2-self.L3**2)/(2*self.L3*l))
        d = np.arcsin(l*np.sin(b2)/self.L3)
        theta[2] = b2-b1+phi-np.pi
        theta[3] = -b1-d+phi-np.pi
        return theta

    def r(self, theta):
        r = np.zeros((4, 3))
        r[0][0] = self.L0*np.cos(theta[0])
        r[0][1] = self.L0*np.sin(theta[0])
        r[0][0] = self.L1*np.cos(theta[1])
        r[0][1] = self.L1*np.sin(theta[1])
        r[0][0] = self.L2*np.cos(theta[2])
        r[0][1] = self.L2*np.sin(theta[2])
        r[0][0] = self.L3*np.cos(theta[3])
        r[0][1] = self.L3*np.sin(theta[3])
        return r

    def omega(self, r, o1z):
        omega = np.zeros((4,3))
        omega[1][2] = o1z
        omaga[2][2] = -o1z*(r[1][1]*r[3][0]-r[3][1]*r[1][0])/(r[2][1]*r[3][0]-r[3][1]*r[2][1])
        omega[3][2] = -o1z*(r[2][1]*r[1][0]-r[1][1]*r[2][0])/(r[2][1]*r[3][0]-r[3][1]*r[2][0])
        return omega

    def alpha(self, r, omega, a1z):
        alpha = np.zeros((4,3))
        alpha[1][2] = a1z
        alpha[2][2] = (
            r[3][0]*(
                -a1z*r[1][1]-omega[1][2]**2*r[1][0]-omega[2][2]**2*r[2][0]-omega[3][3]**2*r[3][0]
            )-r[3][1]*(
                -a1z*r[1][0]-omega[1][2]**2*r[1][2]-omega[2][2]**2*r[2][2]-omega[3][2]**2*r[3][2]
            )
        )/(
            r[2][2]*r[3][0]-r[3][2]*r[2][0]
        )
        alpha[2][2] = (
            -r[2][0]*(
                -a1z*r[1][1]-omega[1][2]**2*r[1][0]-omega[2][2]**2*r[2][0]-omega[3][3]**2*r[3][0]
            )+r[2][1]*(
                -a1z*r[1][0]-omega[1][2]**2*r[1][2]-omega[2][2]**2*r[2][2]-omega[3][2]**2*r[3][2]
            )
        )/(
            r[2][2]*r[3][0]-r[3][2]*r[2][0]
        )
        return alpha

    def v(self, r, omega):
        v = np.zeros((4, 3))
        v[1][0] = -omega[1][2]*r[1][1]
        v[1][1] = omega[1][2]*r[1][0]
        v[2][0] = -omega[2][2]*r[2][1]
        v[2][1] = omega[2][2]*r[2][0]
        v[3][0] = -omega[3][2]*r[3][1]
        v[3][1] = omega[3][2]*r[3][0]
        return v

    def a(self, r, omega, alpha):
        a = np.zeros((4, 3))
        a[1][0] = -alpha[1][2]*r[1][1] - omega[1][2]**2*r[1][0]
        a[1][1] = alpha[1][2]*r[1][0] - omega[1][2]**2*r[1][1]
        a[2][0] = -alpha[2][2]*r[2][1] - omega[2][2]**2*r[2][0]
        a[2][1] = alpha[2][2]*r[2][0] - omega[2][2]**2*r[2][1]
        a[3][0] = -alpha[3][2]*r[3][1] - omega[3][2]**2*r[3][0]
        a[3][1] = alpha[3][2]*r[3][0] - omega[3][2]**2*r[3][1]
        return a

    def poc_kinematics(self, r, theta, omega, alpha, L):
        poc = np.zeros((3, 3))
        poc[0][0] = r[1][0] + L*np.cos(np.pi+theta[2])
        poc[0][1] = r[1][1] + L*np.sin(np.pi+theta[2])
        poc[1][0] = - omega[1][2]*r[1][1] - omega[2][2]*poc[0][1]
        poc[1][1] = omega[1][2]*r[1][0] + omega[2][2]*poc[0][0]
        poc[2][0] = - alpha[1][2]*r[1][1] - omega[1][2]**2*r[1][0] - alpha[2][2]*poc[0][1] - omega[2][2]**2*poc[0][0]
        poc[2][1] = alpha[1][2]*r[1][0] - omega[1][2]**2*r[1][1] + alpha[2][2]*poc[0][0] - omega[2][2]**2*poc[0][1]
        return poc

