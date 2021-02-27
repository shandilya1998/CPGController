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
