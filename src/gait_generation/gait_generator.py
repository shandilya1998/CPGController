import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def get_base_signal(Tsw, Tst, theta, N):
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
def get_signal(dt, Tsw, Tst, N, theta, version = 1):
    T = Tsw+Tst
    out = np.zeros((N, 9))
    t = np.arange(0,N)*dt
    out[:,0] = t
    signal = get_base_signal(Tsw, Tst, theta, N+T)
    """
    Time shifting the signals for obtaining the gait pattern
    """
    if version == 1:
        out = _get_signal(signal, out, T, N)
    elif version == 2:
        out = _get_signal_v2(signal, out, T, N)
    elif version == 3:
        out = _get_signal_v3(signal, out, T, N)
    elif version == 4:
        out = _get_signal_v4(signal, out, T, N)
    else:
        raise ValueError('Expected version 1, 2, 3 or 4, got {ver}'.format(ver = version))
    #save_signal(out, N)
    return out

def _get_signal(signal, out, T, N):
    out[:, 1] = signal[T:, 0]
    out[:, 2] = signal[T:, 1]
    out[:, 3] = signal[T-int(T/4):-int(T/4), 0]
    out[:, 4] = signal[T-int(T/4):-int(T/4), 1]
    out[:, 5] = signal[T-int(T/2):-int(T/2), 0]
    out[:, 6] = signal[T-int(T/2):-int(T/2), 1]
    out[:, 7] = signal[T-int(3*T/4):-int(3*T/4), 0]
    out[:, 8] = signal[T-int(3*T/4):-int(3*T/4), 1]
    return out

def _get_signal_v2(signal, out, T, N):
    out[:, 1] = signal[:N, 0]
    out[:, 2] = signal[:N, 1]
    out[:, 3] = signal[int(T/4):N+int(T/4), 0]
    out[:, 4] = signal[int(T/4):N+int(T/4), 1]
    out[:, 5] = signal[int(T/2):N+int(T/2), 0]
    out[:, 6] = signal[int(T/2):N+int(T/2), 1]
    out[:, 7] = signal[int(3*T/4):N+int(3*T/4), 0]
    out[:, 8] = signal[int(3*T/4):N+int(3*T/4), 1]
    return out

def _get_signal_v3(signal, out, T, N):
    out[:, 1] = signal[T:, 0]
    out[:, 2] = signal[T:, 1]
    out[:, 3] = signal[T-int(T/4):-int(T/4), 0]
    out[:, 4] = signal[T-int(T/4):-int(T/4), 1]
    out[:, 5] = -signal[T-int(T/2):-int(T/2), 0]
    out[:, 6] = signal[T-int(T/2):-int(T/2), 1]
    out[:, 7] = -signal[T-int(3*T/4):-int(3*T/4), 0]
    out[:, 8] = signal[T-int(3*T/4):-int(3*T/4), 1]
    return out

def _get_signal_v4(signal, out, T, N):
    out[:, 1] = signal[:N, 0]
    out[:, 2] = signal[:N, 1]
    out[:, 3] = signal[int(T/4):N+int(T/4), 0]
    out[:, 4] = signal[int(T/4):N+int(T/4), 1]
    out[:, 5] = -signal[int(T/2):N+int(T/2), 0]
    out[:, 6] = signal[int(T/2):N+int(T/2), 1]
    out[:, 7] = -signal[int(3*T/4):N+int(3*T/4), 0]
    out[:, 8] = signal[int(3*T/4):N+int(3*T/4), 1]
    return out
    
def save_signal(out, N, csv = False,):
    fig, axes = plt.subplots(4,1)
    axes[0].plot(out[:2000, 0], out[:2000, 1])
    axes[0].plot(out[:2000, 0], out[:2000, 2]) 
    axes[1].plot(out[:2000, 0], out[:2000, 3])
    axes[1].plot(out[:2000, 0], out[:2000, 4])
    axes[2].plot(out[:2000, 0], out[:2000, 5])
    axes[2].plot(out[:2000, 0], out[:2000, 6])
    axes[3].plot(out[:2000, 0], out[:2000, 7]) 
    axes[3].plot(out[:2000, 0], out[:2000, 8])
    plt.show()
    fig.savefig('four_leg_gait_pattern_9.png')
    #print(list(range(N)))
    df = pd.DataFrame(
        data = out, 
        index = list(range(N)), 
        columns = [
            'time', 
            'hip leg1', 
            'knee leg1', 
            'hip leg4', 
            'knee leg4', 
            'hip leg4', 
            'knee leg4', 
            'hip leg4', 
            'knee leg4'
        ]
    )
    if csv:
        df.to_csv('gait_data.csv', header = True)

#signal = get_signal(0.001, 20, 60, 500, 30, 4)
#save_signal(signal, 500)
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
