import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class Signal:
    def __init__(self, N, dt, pi = np.pi):
        self.N = N
        self.dt = dt
        self.theta1 = 0
        self.theta2 = 0
        self.theta3 = 0
        self.pi = np.pi
        self.Tst = 60
        self.Tsw = 20
        self.theta_h = np.pi/6
        self.theta_k = np.pi/6
        self.build(self.Tst, self.Tsw, self.theta_h, self.theta_k)
        self._get_base()

    def compute_v(self, Lt):
        return (2 * self.theta_h * Lt) / (self.Tsw + self.Tst)

    def build(self, Tsw, Tst, theta_h, theta_k):
        self.theta_h = theta_h
        self.theta_k = theta_k
        self.Tsw = Tsw
        self.Tst = Tst
        self.T = self.Tst + self.Tsw
        self.beta = self.Tst / self.T

    def _compute(self, t):
        t = t % self.T
        if 0 <=t <=self.beta * self.T/2:
            self.theta1 = self.theta_h * np.sin(
                np.pi * t / (
                    self.T * self.beta
                ) + np.pi
            )
            self.theta2 = 0
            self.theta3 = 0
        elif self.T * self.beta / 2 < t < self.T * (2 - self.beta) / 2:
            self.theta1 = self.theta_h * np.sin(
                np.pi * t / (
                    self.T * (1 - self.beta)
                ) + np.pi * (3 - 4 * self.beta)/(2 * (1 - self.beta))
            )
            self.theta2 = self.theta_k * np.sin(
                np.pi * t/(
                    self.T * (1 - self.beta)
                ) - np.pi * self.beta/(2 * (1 - self.beta))
            )
            self.theta3 = -self.theta2 / 5
        elif self.T * (2 - self.beta) / 2 <= t < self.T:
            self.theta1 = self.theta_h * np.sin(
                np.pi * t / (
                    self.T * self.beta
                ) + np.pi * (self.beta - 1) / self.beta
            )
            self.theta2 = 0
            self.theta3 = 0


    def _get_base(self):
        self.base_signal = np.zeros((self.N + 2*self.T, 3))
        for i in range(self.N + 2 * self.T):
            self._compute(i)
            self.base_signal[i, 0] = self.theta1
            self.base_signal[i, 1] = self.theta2
            self.base_signal[i, 2] = self.theta3
        return self.base_signal

    def get_signal(self):
        signal = self._get_base()
        out = np.zeros((self.N, 13))
        t = np.arange(0, self.N) * self.dt
        out[:, 0] = t
        phases = []
        t = int(self.T + self.Tst / 4 - self.Tsw / 4)
        out[:, 3] = signal[t : self.N + t, 0]
        out[:, 2] = signal[t : self.N + t, 1]
        out[:, 1] = signal[t : self.N + t, 2]
        phases.append(0)
        t = int(self.T - self.Tst / 2)
        out[:, 6] = -signal[t : self.N + t, 0]
        out[:, 5] = signal[t : self.N + t, 1]
        out[:, 4] = signal[t : self.N + t, 2]
        phases.append(0.75)
        t = int(self.T + self.Tst / 2)
        out[:, 9] = signal[t : self.N + t, 0]
        out[:, 8] = signal[t : self.N + t, 1]
        out[:, 7] = signal[t : self.N + t, 2]
        phases.append(-0.25)
        t = int(self.T + self.Tsw/4 - self.Tst/4)
        out[:, 12] = -signal[t : self.N + t, 0]
        out[:, 11] = signal[t : self.N + t, 1]
        out[:, 10] = signal[t : self.N + t, 2]
        phases.append(0.25)
        return out, phases

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
        out, phases = _get_signal(signal, out, T, N)
    elif version == 2:
        out, phases = _get_signal_v2(signal, out, T, N)
    elif version == 3:
        out, phases = _get_signal_v3(signal, out, T, N)
    elif version == 4:
        out, phases = _get_signal_v4(signal, out, T, N)
    elif version == 5:
        out, phases = _get_signal_v5(signal, out, T, N)
    elif version == 6:
        out, phases = _get_signal_v6(signal, out, T, N)
    elif version == 7:
        out, phases = _get_signal_v7(signal, out, T, N)
    elif version == 8:
        out, phases = _get_signal_v8(signal, out, T, N)
    elif version == 9:
        out, phases = _get_signal_v9(signal, out, T, N)
    else:
        raise ValueError('Expected version 1, 2, 3 or 4, got {ver}'.format(ver = version))
    #save_signal(out, N)
    return out, phases

def _get_signal(signal, out, T, N):
    phases = []
    out[:, 1] = signal[T:, 0]
    out[:, 2] = signal[T:, 1]
    phases.append(0)
    out[:, 3] = signal[T-int(T/4):-int(T/4), 0]
    out[:, 4] = signal[T-int(T/4):-int(T/4), 1]
    phases.append(0.25)
    out[:, 5] = signal[T-int(T/2):-int(T/2), 0]
    out[:, 6] = signal[T-int(T/2):-int(T/2), 1]
    phases.append(0.5)
    out[:, 7] = signal[T-int(3*T/4):-int(3*T/4), 0]
    out[:, 8] = signal[T-int(3*T/4):-int(3*T/4), 1]
    phases.append(0.75)
    return out, phases

def _get_signal_v2(signal, out, T, N):
    phases = []
    out[:, 1] = signal[:N, 0]
    out[:, 2] = signal[:N, 1]
    phases.append(0)
    out[:, 3] = signal[int(T/4):N+int(T/4), 0]
    out[:, 4] = signal[int(T/4):N+int(T/4), 1]
    phases.append(-0.25)
    out[:, 5] = signal[int(T/2):N+int(T/2), 0]
    out[:, 6] = signal[int(T/2):N+int(T/2), 1]
    phases.append(-0.5)
    out[:, 7] = signal[int(3*T/4):N+int(3*T/4), 0]
    out[:, 8] = signal[int(3*T/4):N+int(3*T/4), 1]
    phases.append(-0.75)
    return out, phases

def _get_signal_v3(signal, out, T, N):
    phases = []
    out[:, 1] = signal[T:, 0]
    out[:, 2] = signal[T:, 1]
    phases.append(0)
    out[:, 3] = signal[T-int(T/4):-int(T/4), 0]
    out[:, 4] = signal[T-int(T/4):-int(T/4), 1]
    phases.append(0.25)
    out[:, 5] = -signal[T-int(T/2):-int(T/2), 0]
    out[:, 6] = signal[T-int(T/2):-int(T/2), 1]
    phases.append(0.5)
    out[:, 7] = -signal[T-int(3*T/4):-int(3*T/4), 0]
    out[:, 8] = signal[T-int(3*T/4):-int(3*T/4), 1]
    phases.append(0.75)
    return out, phases

def _get_signal_v4(signal, out, T, N):
    phases = []
    out[:, 1] = signal[:N, 0]
    out[:, 2] = signal[:N, 1]
    phases.append(0)
    out[:, 3] = signal[int(T/4):N+int(T/4), 0]
    out[:, 4] = signal[int(T/4):N+int(T/4), 1]
    phases.append(-0.25)
    out[:, 5] = -signal[int(T/2):N+int(T/2), 0]
    out[:, 6] = signal[int(T/2):N+int(T/2), 1]
    phases.append(-0.5)
    out[:, 7] = -signal[int(3*T/4):N+int(3*T/4), 0]
    out[:, 8] = signal[int(3*T/4):N+int(3*T/4), 1]
    phases.append(-0.75)
    return out, phases

def _get_signal_v5(signal, out, T, N):
    """
        Trot Gait
    """
    phases = []
    out[:, 1] = signal[:N, 0]
    out[:, 2] = signal[:N, 1]
    phases.append(0)
    out[:, 3] = signal[int(T/2):N+int(T/2), 0]
    out[:, 4] = signal[int(T/2):N+int(T/2), 1]
    phases.append(-0.5)
    out[:, 5] = -signal[:N, 0]
    out[:, 6] = signal[:N, 1]
    phases.append(0)
    out[:, 7] = -signal[int(T/2):N+int(T/2), 0]
    out[:, 8] = signal[int(T/2):N+int(T/2), 1]
    phases.append(-0.5)
    return out, phases  
        
def _get_signal_v6(signal, out, T, N):
    """ 
        Pace Gait
    """     
    phases = []
    out[:, 1] = signal[:N, 0]
    out[:, 2] = signal[:N, 1]
    phases.append(0)
    out[:, 3] = signal[int(T/2):N+int(T/2), 0]
    out[:, 4] = signal[int(T/2):N+int(T/2), 1]
    phases.append(-0.5)
    out[:, 5] = -signal[int(T/2):N+int(T/2), 0]
    out[:, 6] = signal[int(T/2):N+int(T/2), 1]
    phases.append(-0.5)
    out[:, 7] = -signal[:N, 0]
    out[:, 8] = signal[:N, 1]
    phases.append(0)
    return out, phases

def _get_signal_v7(signal, out, T, N):
    """ 
        Bound Gait
    """
    phases = []
    out[:, 1] = signal[:N, 0]
    out[:, 2] = signal[:N, 1]
    phases.append(0)
    out[:, 3] = signal[:N, 0]
    out[:, 4] = signal[:N, 1]
    phases.append(0)
    out[:, 5] = -signal[int(T/2):N+int(T/2), 0]
    out[:, 6] = signal[int(T/2):N+int(T/2), 1]
    phases.append(-0.5)
    out[:, 7] = -signal[int(T/2):N+int(T/2), 0]
    out[:, 8] = signal[int(T/2):N+int(T/2), 1]
    phases.append(-0.5)
    return out, phases

def _get_signal_v8(signal, out, T, N):
    """ 
        Transverse Gallop Gait
    """
    phases = []
    out[:, 1] = signal[:N, 0]
    out[:, 2] = signal[:N, 1]
    phases.append(0)
    out[:, 3] = signal[int(T/10):N+int(T/10), 0]
    out[:, 4] = signal[:N, 1]
    phases.append(-0.1)
    out[:, 5] = -signal[int(3*T/5):N+int(3*T/5), 0]
    out[:, 6] = signal[int(3*T/5):N+int(3*T/5), 1]
    phases.append(-0.6)
    out[:, 7] = -signal[int(T/2):N+int(T/2), 0]
    out[:, 8] = signal[int(T/2):N+int(T/2), 1]
    phases.append(-0.5)
    return out, phases

def _get_signal_v9(signal, out, T, N):
    """ 
        Rotary Gallop Gait
    """
    phases = []
    out[:, 1] = signal[:N, 0]
    out[:, 2] = signal[:N, 1]
    phases.append(0)
    out[:, 3] = signal[int(T/10):N+int(T/10), 0]
    out[:, 4] = signal[int(T/10):N+int(T/10), 1]
    phases.append(-0.1)
    out[:, 5] = -signal[int(T/2):N+int(T/2), 0]
    out[:, 6] = signal[int(T/2):N+int(T/2), 1]
    phases.append(-0.5)
    out[:, 7] = -signal[int(3*T/5):N+int(3*T/5), 0]
    out[:, 8] = signal[int(3*T/5):N+int(3*T/5), 1]
    phases.append(-0.6)
    return out, phases
 
def save_signal(out, N, figname, phases, csv = False, csv_name = None):
    fig = plt.figure(figsize = (12, 12))
    i = 1
    while(i<9):
        axis = fig.add_subplot(4,1,(i+1)/2)
        axis.plot(out[:2000, 0], out[:2000, i], 'b', label = 'hip activation')
        axis.plot(out[:2000, 0], out[:2000, i+1], 'r', label = 'knee activation')
        axis.legend()
        axis.set_title('Activation relative phase {phase}'.format(phase = phases[int((i-1)/2)]))
        axis.set_xlabel('time(s)')
        axis.set_ylabel('activation(degrees)') 
        i+=2
    fig.tight_layout(pad = 2.0)
    plt.show()
    fig.savefig('../../images/gait_activations/{name}'.format(name=figname))
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
        if csv_name ==None:
            raise ValueError('Expected a string value for csv file name, got None')
        else:
            df.to_csv('gait_data_3.csv', header = True)

def _support_pos(signals, N):
    pos = np.zeros((N, 9))
    pos[:,0] = signals[:,0]
    extremes = []
    pos[:, 1] = 45.25 + (37.50 + 27.869*np.cos(np.pi*signals[:, 2]/180))*np.cos(np.pi*signals[:, 1]/180 + np.pi/4)
    pos[:, 2] = 45.25 + (37.50 + 27.869*np.cos(np.pi*signals[:, 2]/180))*np.sin(np.pi*signals[:, 1]/180 + np.pi/4)
    pos[:, 3] = 45.25 + (37.50 + 27.869*np.cos(np.pi*signals[:, 4]/180))*np.cos(np.pi*signals[:, 3]/180 - np.pi/4)
    pos[:, 4] = -45.25 + (37.50 + 27.869*np.cos(np.pi*signals[:, 4]/180))*np.sin(np.pi*signals[:, 3]/180 - np.pi/4)
    pos[:, 5] = -45.25 + (37.50 + 27.869*np.cos(np.pi*signals[:, 6]/180))*np.cos(np.pi*signals[:, 5]/180 - 3*np.pi/4)
    pos[:, 6] = -45.25 + (37.50 + 27.869*np.cos(np.pi*signals[:, 6]/180))*np.sin(np.pi*signals[:, 5]/180 - 3*np.pi/4)
    pos[:, 7] = -45.25 + (37.50 + 27.869*np.cos(np.pi*signals[:, 8]/180))*np.cos(np.pi*signals[:, 7]/180 + 3*np.pi/4)
    pos[:, 8] = 45.25 + (37.50 + 27.869*np.cos(np.pi*signals[:, 8]/180))*np.sin(np.pi*signals[:, 7]/180 + 3*np.pi/4)
    return pos


def support_polygon(dt, Tsw, Tst, N, theta, version = 4, figname = 'support_polygon_trajectory.png'):
    signals, phases = get_signal(dt, Tsw, Tst, N, theta, version)
    pos = _support_pos(signals, N)
    fig, axes = plt.subplots(1,1, figsize = (10,10))
    axes.plot(pos[:, 0], np.sqrt(np.square((pos[:, 1]-pos[:,3])) + np.square((pos[:,2] - pos[:,4]))))
    axes.set_xlabel('time')
    axes.set_ylabel('end effector joining line length')
    axes.set_title('Support Polygon')
    plt.show()
    fig.savefig('../../images/gait_activations/{fig}'.format(fig=figname))

def support_lines(dt, Tsw, Tst, N, theta, version = 0, figname = 'support_pos_trajectory.png'):
    """
        All calculations are assuming that the robot body remains parallel to the ground while traversing
    """
    signals, phases = get_signal(dt, Tsw, Tst, N, theta, version)
    pos = _support_pos(signals, N)
    fig, axes = plt.subplots(1, 1, figsize = (10, 10))
    axes.plot(pos[: 5*(Tsw+Tst), 1], pos[: 5*(Tsw+Tst), 2])
    axes.plot(pos[: 5*(Tsw+Tst), 3], pos[: 5*(Tsw+Tst), 4])
    axes.plot(pos[: 5*(Tsw+Tst), 5], pos[: 5*(Tsw+Tst), 6])
    axes.plot(pos[: 5*(Tsw+Tst), 7], pos[: 5*(Tsw+Tst), 8])
    axes.set_xlabel('x coordinate')
    axes.set_ylabel('y coordinate')
    axes.set_title('Support Trajectory Leg')
    fig.tight_layout(pad = 0.75)
    plt.show()
    fig.savefig('../../images/gait_activations/{fig}'.format(fig=figname))
    extremes = []
    for i in range(4):
        extremes.append(
            [
                [
                    np.amax(pos[:, 2*i+1]), 
                    np.amax(pos[: 2*i+2])
                ],
                [
                    np.amin(pos[:, 2*i+1]),
                    np.amin(pos[:, 2*i+2])
                ]    
            ] 
        )
"""
These values dt = 0.001, Tsw = 20, Tst = 60, N = 500, theta = 30 are used
for experiments 6 through 8
"""
"""
dt = 0.001
Tsw = 80
Tst = 200
N = 5000
theta = 75
version = 4
gait = 'creep'
figname = 'support_pos_trajectory_{g}_gait.png'.format(g = gait)
signal, phases = get_signal(dt, Tsw, Tsw, N, theta, version)
support_polygon(dt, Tsw, Tst, N, theta, version, figname)
#save_signal(signal, 50000, 'rotary_gallop_gait_tst80_tsw80_dt10e-3_theta45.png', phases, False)
"""
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
