from reward.zmp import ZMP
import numpy as np

class FitnessFunction:
    def __init__(self, params):
        self.params = params
        self.zmp = ZMP(params)

    def build(self, t, Tb, A, B, AL, BL, AF, BF):
        self.A = A
        self.AL = AL
        self.AF = AF
        self.B = B
        self.BL = BL
        self.BF = BF
        self.zmp.build(t, Tb, A, B, AL, BL, AF, BF)

    def __call__(self, com, force, torque):
        zmp_s = self.zmp(com, force, torque)
        return None
