import numpy as np

class SupportPlane:
    def __init__(self, tb = 1000):
        self.A = np.zeros(3)
        self.AL = np.zeros(3)
        self.AF = np.zeros(3)
        self.B = np.zeros(3)
        self.BL = np.zeros(3)
        self.BF = np.zeros(3)
        self.AB = self.B - self.A
        self.Tb = tb

    def build(self, A, B, AL, AF, BF, BL):
        self.A = A
        self.AL = AL
        self.AF = AF
        self.B = B
        self.BL = BL
        self.BF = BF
        self.AB = self.B - self.A        

    def set_tb(self, tb):
        self.Tb = tb

    def get_n11(self):
        AAf = self.AF-self.A
        cross = np.cross(self.AB, AAf)
        return cross/np.linalg.norm(cross)

    def get_n12(self):
        BBf = self.BF - self.B
        cross = np.cross(self.AB, BBf)
        return cross/np.linalg.norm(cross)

    def get_n21(self):
        AAl = self.AL - self.A
        cross = np.cross(self.AB, AAl)
        return cross/np.linal.norm(cross)

    def get_n22(self):
        BBl = self.BL - self.B
        cross = np.cross(self.AB, BBl)
        return cross/np.linalg.norm(cross)

    def get_n1(self):
        n11 = self.get_n11()
        n12 = self.get_n12()
        return (n11 + n12)/np.linalg.norm(n11 + n12)

    def get_n2(self):
        n21 = self.get_n21()
        n22 = self.get_n22()
        return (n21 + n22)/np.linalg.norm(n21 + n22)

    def get_xs(self, t):
        mu = -t/self.Tb + 1
        n1 = self.get_n1()
        n2 = self.get_n2()
        temp = mu*n1 + (1-mu)*n2
        return temp/np.linalg.norm(temp)

    def get_zs(self):
        return self.AB/np.linalg.norm(self.AB)

    def get_ys(self, t, xs = None, zs = None):
        if xs != None and zs != None:
            return np.cross(zs, xs)

        elif xs == None:
            xs = self.get_xs(t)
            return np.cross(zs, xs)

        elif zs == None:
            zs = self.get_zs()
            return np.cross(zs, xs)

        else:
            xs = self.get_xs(t)
            zs = self.get_zs()
            return np.cross(zs, xs)

    def get_support_plane(self, t):
        xs = self.get_xs(t)
        zs = self.get_zs()
        ys = self.get_ys(t, xs, zs)
        plane = np.zeros((3, 3))
        plane[0, :] = xs/np.norm(xs)
        plane[1, :] = ys/np.norm(ys)
        plane[2, :] = zs/np.norm(zs)
        return plane
