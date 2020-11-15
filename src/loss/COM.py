import numpy as np

class COM:
    """
        All units are in grams
        The default values are assuming the origin at the position of the center of mass when 
        the quadruped is at rest with the leg parallel linkage orthogonal to leg rev
    """
    def __init__(self, L = None, M = None, m = None):
        if L == None:
            self.L = np.array(
                [
                    [27.5394, 26.9089, 0.3235],
                    [-27.5394, 26.9089, 0.3235],
                    [-27.5394, -26.9089, 0.3235],
                    [27.5394, -26.9089, 0.3235]
                ]
            )
    
        if M == None:
            self.M = 65.15

        if m == None:
            self.m = 7.707

    def __call__(self, theta):
        com_base = np.zeros(3)
        pos = np.zeros(4, 3)
        pos[0][0] = np.cos(theta[0])
        pos[0][1] = np.sin(theta[0])
        pos[0][2] = 1
        pos[1][0] = np.cos(theta[1])
        pos[1][1] = np.sin(theta[1])
        pos[0][2] = 1 
        pos[2][0] = np.cos(theta[2])
        pos[2][1] = np.sin(theta[2])
        pos[0][2] = 1 
        pos[3][0] = np.cos(theta[3])
        pos[3][1] = np.sin(theta[3])
        pos[0][2] = 1 
        return self.m*np.sum(self.L*pos, 0)/self.M
