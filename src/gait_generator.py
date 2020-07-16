import numpy as np
import matplotlib.pyplot as plt

def get_gait(Tsw, Tst, theta, N):
    signal = np.zeros((N,2))
    T = Tst+Tsw
    beta = Tst/T
    for i in range(N):
        t = i%T
        if 0<=t<=beta*T/2:
            signal[i][0] = theta*np.sin(np.pi*t/(T*beta)+np.pi)
        elif T*beta/2 < t < T*(2-beta)/2:
            signal[i][0] = theta*np.sin(np.pi*t/(T*(1-beta)) + np.pi*(3-4*beta)/(2*(1-beta)))
            signal[i][1]= theta*np.sin(np.pi*t/(T*(1-beta)) - np.pi*beta/(2*(1-beta)))
        elif T*(2-beta)/2 <= t < T:
            signal[i][0] = theta*np.sin(np.pi*t/(T*beta)+np.pi*(beta-1)/beta)
    return signal

Tsw = 10
Tst = 30
T = Tsw+Tst
N = 1000
theta = 15
signal = get_gait(Tsw, Tst, theta, N)
signal1_h = signal[T:200+T,0]
signal1_k = signal[:200,1]
signal2_h = signal[T-int(T/2):T+200-int(T/2),0]
signal2_k = signal[T-int(T/2):T+200-int(T/2),1]
signal3_h = signal[T-int(T/4):T+200-int(T/4),0]
signal3_k = signal[T-int(T/4):T+200-int(T/4),1]
signal4_h = signal[T-int(3*T/4):T+200-int(3*T/4),0]
signal4_k = signal[T-int(3*T/4):T+200-int(3*T/4),1]
t = np.arange(0,N)
fig, axes = plt.subplots(4,1)
axes[0].plot(t[:200], signal1_h)
axes[0].plot(t[:200], signal1_k) 
axes[1].plot(t[:200], signal2_h)
axes[1].plot(t[:200], signal2_k)
axes[2].plot(t[:200], signal3_h)
axes[2].plot(t[:200], signal3_k)
axes[3].plot(t[:200], signal4_h) 
axes[3].plot(t[:200], signal4_k)
plt.show()
fig.savefig('four_leg_gait_pattern_3.png')
"""
    Experiment 0
        Tsw = 15
        Tst = 15
        theta = 15      
        factors of T/4 used to time shift the signal
    Experiment 1
        Tsw = 10
        Tst = 30
        theta = 15
        factors of T/4 used to time shift the signal
    In the previous experiments the signals were left shifted onto the x-axis
    Experiment 2
        Tsw = 10
        Tst = 30
        theta = 15
        factors of T/4 used to time shift the signal
        signals are now right shifted
    Experiment 3
        Tsw = 10
        Tst = 30
        theta = 15
        factors of T/4 used to time shift the signal
        signals are now right shifted
        this experiment simulates the ideal gait pattern for a LS walk, while previous were DS walk
"""
