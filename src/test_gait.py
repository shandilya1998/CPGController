from learn import *
from rl.constants import params
import numpy as np
from gait_generation import gait_generator as gg
import matplotlib.pyplot as plt

params['rnn_steps'] = 10000
signal_gen = gg.Signal(params['rnn_steps'], params['dt'])

signal_gen.build(100, 300, 45, 30)
signal, phases = signal_gen.get_signal()
t = signal[:, 0]
signal = signal[:, 1:] * np.pi/180

fig, axes = plt.subplots(4, 1, figsize = (20,20))
legs = ['Front Right', 'Front Left', 'Back Right', 'Back Left']

for i in range(4):
    axes[i].plot(t[:1000], signal[:1000, 3 * i], 'r', label = 'Ankle')
    axes[i].plot(t[:1000], signal[:1000, 3 * i + 1], 'b', label = 'Knee')
    axes[i].plot(t[:1000], signal[:1000, 3 * i + 2], 'g', label = 'Hip')
    axes[i].set_title(legs[i])
    axes[i].legend(loc='upper left', frameon=False)

fig.savefig('gait_pattern.png')


learner = Learner(params, False)
"""
dataset = learner.load_dataset()
step,(x,y) = next(enumerate(dataset))


action = [y[0].numpy() * np.pi/180, y[3].numpy()]
desired_motion = x[0][0].numpy()

y = y[0].numpy() * np.pi / 180
print(y.shape)

osc = y[3]
print(osc.shape)
"""
#"""
for i in range(params['rnn_steps']):
    learner.env.quadruped.all_legs.move(
            signal[i].tolist()
    )
#"""
"""
from gait_generation import gait_generator as gg
import matplotlib.pyplot as plt
signal_gen = gg.Signal(500, 0.001)
signal_gen.build(30, 75, 45, 45)
signal, phases = signal_gen.get_signal()
t = signal[:, 0]
signal = signal[:, 1:]
fig, axes = plt.subplots(4,1 ,figsize = (15,15))
for i in range(4):
    axes[i].plot(t, signal[:, 3*i], 'r')
    axes[i].plot(t, signal[:, 3*i+1], 'b')
    axes[i].plot(t, signal[:, 3*i+2], 'g')
fig.savefig('data/pretrain/gait_pattern2.png')
"""
