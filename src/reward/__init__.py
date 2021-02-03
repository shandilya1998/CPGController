from reward.zmp import ZMP
import numpy as np

class FitnessFunction:
    def __init__(self, params):
        self.params = params
        self.zmp = ZMP(params)

    def build(self, t, Tb, A, B, AL, BL, AF, BF):
        self.t = t
        self.Tb = Tb
        self.A = A
        self.AL = AL
        self.AF = AF
        self.B = B
        self.BL = BL
        self.BF = BF
        self.zmp.build(t, Tb, A, B, AL, BL, AF, BF)

    def __call__(self, com, force, torque, v_real, v_exp, eta, omega):
        zmp = self.zmp(com, force, torque, v_real, v_exp, eta)
        dc = np.abs(np.cross(
            (self.A - zmp),
            (self.A - self.B)
        ) / np.norm((self.A - self.B)))
        dl = np.abs(np.cross(
            (self.BL - zmp),
            (self.AL - self.BL)
        ) / np.norm(self.AL - self.BL))
        wc = 0
        if 0.25*self.Tb < self.t < self.Tb*0.75:
            wc = -2*self.t/self.Tb + 1.5
        elif 0 < self.t < 0.25 * self.Tb:
            wc = 1
        wl = 1
        if 0.25*self.Tb < self.t < self.Tb*0.75:
            wc = 2*self.t/self.Tb - 1.5
        elif 0 < self.t < 0.25 * self.Tb:
            wc = 0
        d_spt = wc * dc + wl * dl

        d11 = np.abs(np.cross(
            (self.A - zmp),
            (self.A - self.BL)
        ) / np.norm((self.A - self.BL)))
        d12 = np.abs(np.cross(
            (self.AL - zmp),
            (self.AL - self.B)
        ) / np.norm((self.AL - self.B)))
        d21 = np.abs(np.cross(
            (self.A - zmp),
            (self.A - self.AL)
        ) / np.norm((self.A - self.AL)))
        d22 = np.abs(np.cross(
            (self.B - zmp),
            (self.BL - self.B)
        ) / np.norm((self.BL - self.B)))
        d_edge = omega*(min(d11, d12) + min(d21, d22))

        u = zmp - com
        cosT = np.dot(u, self.plane[0])/(np.norm(u) * np.norm(self.plane[0]))
        sinT = np.sqrt(1-cosT*cosT)

        stability = d_edge - \
            ((self.params['L'] + self.params['W']) / 8) * sinT - \
            ((self.params['L'] + self.params['W']) * \
                0.9 / (self.params['W'] * 4 )) * d_spt



        return 0
