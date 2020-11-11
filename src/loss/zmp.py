import numpy as np

class ZMP:
    def __init__(self, mass = 0.300, gravity = 10):
        self.m = mass
        self.g = gravity

    def get_N(self, legs):
        """
            legs : 
                type : np.array(type = np.int)
                shape : (4, )
                description : 
                    legs[i] = 0 if leg i is not in contact with the ground
                    legs[i] = 1 if leg i is in contact with the ground
        """
        
        
