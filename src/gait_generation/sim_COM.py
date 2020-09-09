import numpy as np
import matplotlib.pyplot as plt
import gait_generator as gg

def sim_COM(Tsw, Tst, theta, N):
    signal = gg.get_signal(Tsw, Tst, theta, N)
    
