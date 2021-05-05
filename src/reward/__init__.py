from reward.zmp import ZMP
import numpy as np

class FitnessFunction:
    def __init__(self, params):
        self.params = params
        self.zmp = ZMP(params)

    def build(self, t, Tb, A, B, AL, BL, AF, BF):
        self.t = t
        self.Tb = Tb
        self.A = A['position']
        self.AL = AL['position']
        self.AF = AF['position']
        self.B = B['position']
        self.BL = BL['position']
        self.BF = BF['position']
        self.zmp.build(t, Tb, A, B, AL, BL, AF, BF)

    def stability_reward(self, com, force, torque, v_real, v_exp, eta, \
            mass, g):
        zmp = self.zmp(com, force, torque, v_real, v_exp, eta)
        dc = np.abs(np.cross(
            (self.A - zmp),
            (self.A - self.B)
        ) / np.linalg.norm((self.A - self.B)))

        dl = np.abs(np.cross(
            (self.BL - zmp),
            (self.AL - self.BL)) / np.linalg.norm(self.AL - self.BL)
        )
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


        d11 = np.linalg.norm(np.cross(
            (self.A - zmp),
            (self.A - self.BL)
        ) / np.linalg.norm((self.A - self.BL)))
        d12 = np.linalg.norm(np.cross(
            (self.AL - zmp),
            (self.AL - self.B)
        ) / np.linalg.norm((self.AL - self.B)))
        d21 = np.linalg.norm(np.cross(
            (self.A - zmp),
            (self.A - self.AL)
        ) / np.linalg.norm((self.A - self.AL)))
        d22 = np.linalg.norm(np.cross(
            (self.B - zmp),
            (self.BL - self.B)
        ) / np.linalg.norm((self.BL - self.B)))
        d_edge = (min(d11, d12) + min(d21, d22))
        u = zmp - com
        cosT = np.dot(u, self.zmp.plane[0])/(np.linalg.norm(u) * np.linalg.norm(self.zmp.plane[0]))
        sinT = np.sqrt(1-cosT*cosT)

        d1 = d_edge
        d2 = ((self.params['L'] + self.params['W']) / 8) * sinT
        d3 = np.sum(((self.params['L']+self.params['W'])*0.9/(self.params['W']*4))*d_spt)
        if np.isnan(d1):
            d1 = -1.0
        if np.isnan(d2):
            d2 = 1.0
        if np.isnan(d3):
            d3 = 1.0
        stability = np.sum(d1 - d2 - d3)
        return d1, d2, d3, stability

    def COT(self, joint_torque, joint_vel, v_real, mass, g, dt):
        p_e = np.sum(np.abs(joint_torque * joint_vel))
        if p_e < 0:
            p_e = 0
        #P = np.sum(joint_torque * 5.0/472.22) + p_e
        #COT = P/(mass * np.linalg.norm(g) * np.linalg.norm(v_real))
        COT = p_e * dt
        return -1 * COT

    def motion_reward(self, pos, last_pos, desired_motion):
        motion = np.dot(pos - last_pos, desired_motion[:3])
        return motion
