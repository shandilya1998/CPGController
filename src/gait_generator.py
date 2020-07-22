import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_gait(Tsw, Tst, theta, N):
    """
        Method to generate the common signal
    """
    signal = np.zeros((N, 2))
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


"""
    The beta for an ideal walking gait should be 1/4. 
    Any value more than this would result into multiple legs moving up at the same time
"""
dt = 0.001
Tsw = 100
Tst = 300
T = Tsw+Tst
N = 100000
theta = 15
out = np.zeros((N, 9))
t = np.arange(0,N)*dt
out[:,0] = t
signal = get_gait(Tsw, Tst, theta, N+T)
"""
    Time shifting the signals for obtaining the gait pattern
"""
out[:, 1] = signal[T:, 0]
out[:, 2] = signal[T:, 1]
out[:, 3] = signal[T-int(T/4):-int(T/4), 0]
out[:, 4] = signal[T-int(T/4):-int(T/4), 1]
out[:, 5] = signal[T-int(T/2):-int(T/2), 0]
out[:, 6] = signal[T-int(T/2):-int(T/2), 1]
out[:, 7] = signal[T-int(3*T/4):-int(3*T/4), 0]
out[:, 8] = signal[T-int(3*T/4):-int(3*T/4), 1]
fig, axes = plt.subplots(4,1)
axes[0].plot(t[:2000], out[:2000, 1])
axes[0].plot(t[:2000], out[:2000, 2]) 
axes[1].plot(t[:2000], out[:2000, 3])
axes[1].plot(t[:2000], out[:2000, 4])
axes[2].plot(t[:2000], out[:2000, 5])
axes[2].plot(t[:2000], out[:2000, 6])
axes[3].plot(t[:2000], out[:2000, 7]) 
axes[3].plot(t[:2000], out[:2000, 8])
plt.show()
fig.savefig('four_leg_gait_pattern_3.png')
#print(list(range(N)))
df = pd.DataFrame(data = out, index = list(range(N)), columns = ['time', 'hip leg1', 'knee leg1', 'hip leg4', 'knee leg4', 'hip leg4', 'knee leg4', 'hip leg4', 'knee leg4'])
df.to_csv('gait_data.csv', header = True)
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
