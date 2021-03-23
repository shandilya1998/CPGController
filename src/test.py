from rl.net import ActorNetwork
from rl.constants import params
from rl.env import Env
import tf_agents as tfa
import tensorflow as tf
from learn import SignalDataGen
import numpy as np
from gait_generation.gait_generator import Signal
from frequency_analysis import frequency_estimator
import rospy
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
"""
actor = ActorNetwork(params)
time_step_spec = tfa.trajectories.time_step.time_step_spec(
    observation_spec = params['observation_spec'],
    reward_spec = params['reward_spec']
)
env = Env(time_step_spec, params)

S = env.quadruped.get_state_tensor()

optimizer = tf.keras.optimizers.Adam(
    learning_rate = params['LRA']
)
signal_gen = SignalDataGen(params)

y, x = next(signal_gen.generator())
env.quadruped.reset()
env.quadruped.set_initial_motion_state(x)
S = env.quadruped.get_state_tensor()
with tf.GradientTape() as tape:
    y_pred = actor.model(S)
    loss = tf.keras.losses.mean_squared_error(y, y_pred[0])
grads = tape.gradient(loss, actor.model.trainable_variables)
print(grads)
optimizer.apply_gradients(zip(grads, actor.model.trainable_variables))
"""

dt = params['dt']
def _hopf_oscillator( omega, mu, b, z):
    rng = np.arange(1, params['units_osc'] + 1)
    x, y = z[:params['units_osc']], z[params['units_osc']:]
    x = x + ((mu - (x*x + y*y)) * x - omega * rng * y) * dt + b
    y = y + ((mu - (x*x + y*y)) * y + omega * rng * x) * dt + b
    return np.concatenate([x, y], -1)

time_step_spec = tfa.trajectories.time_step.time_step_spec(
    observation_spec = params['observation_spec'],
    reward_spec = params['reward_spec']
)

env = Env(time_step_spec, params)
signal_gen = Signal(3501, dt)
signal_gen.build(30, 90, 45, 30 )
y, _ = signal_gen.get_signal()
y = y[:, 1:] * np.pi/180
v = signal_gen.compute_v((0.1+0.015)*2.2)
desired_motion = np.array([1, 0, 0, v, 0 ,0], dtype = np.float32)
mu = np.array([30, 30 / 5, 45])
mu = [mu for i in range(4)]
f = 2 * np.pi * frequency_estimator.freq_from_autocorr(y[:, 0], dt)

fig, axes = plt.subplots(4, 1, figsize = (20,20))
legs = ['Front Right', 'Front Left', 'Back Right', 'Back Left']

t = np.arange(y.shape[0]) * dt
for i in range(4):
    axes[i].plot(t[:1000], y[:1000, 3 * i], 'r', label = 'Ankle')
    axes[i].plot(t[:1000], y[:1000, 3 * i + 1], 'b', label = 'Knee')
    axes[i].plot(t[:1000], y[:1000, 3 * i + 2], 'g', label = 'Hip')
    axes[i].set_title(legs[i])
    axes[i].legend(loc='upper left', frameon=False)

fig.savefig('gait_pattern.png')

y = tf.convert_to_tensor(np.expand_dims(y, 0))

current_time_step = env.reset()
rospy.sleep(1.0)
motion_state, robot_state, osc_state = env.quadruped.get_state_tensor()
osc = np.zeros((1, 2*params['units_osc']))
osc = tf.convert_to_tensor(osc)
print(y.shape)
rewards = []
deltas = []
tb = []
d1 = []
d2 = []
d3 = []
mu = []
ALx = []
ALy = []
ALz = []
Ax = []
Ay = []
Az = []
AFx = []
AFy = []
AFz = []
BLx = []
BLy = []
BLz = []
Bx = []
By = []
Bz = []
BFx = []
BFy = []
BFz = []
comx = []
comy = []
comz = []
for i in tqdm(range(1500)):
    signal = y[:, i : i + params['rnn_steps'], :]
    current_time_step=env.step([signal,osc], desired_motion)
    rewards.append(current_time_step.reward)
    deltas.append(env.quadruped.delta)
    tb.append(env.quadruped.Tb)
    d1.append(np.sum(env.quadruped.d1))
    d2.append(np.sum(env.quadruped.d2))
    d3.append(np.sum(env.quadruped.d3))
    mu.append(-env.quadruped.t/env.quadruped.Tb + 1)
    ALx.append(env.quadruped.AL['position'][0])
    ALy.append(env.quadruped.AL['position'][1])
    ALz.append(env.quadruped.AL['position'][2])
    Ax.append(env.quadruped.A['position'][0])
    Ay.append(env.quadruped.A['position'][1])
    Az.append(env.quadruped.A['position'][2])
    AFx.append(env.quadruped.AF['position'][0])
    AFy.append(env.quadruped.AF['position'][1])
    AFz.append(env.quadruped.AF['position'][2])
    BLx.append(env.quadruped.BL['position'][0])
    BLy.append(env.quadruped.BL['position'][1])
    BLz.append(env.quadruped.BL['position'][2])
    Bx.append(env.quadruped.B['position'][0])
    By.append(env.quadruped.B['position'][1])
    Bz.append(env.quadruped.B['position'][2])
    BFx.append(env.quadruped.BF['position'][0])
    BFy.append(env.quadruped.BF['position'][1])
    BFz.append(env.quadruped.BF['position'][2])
    comx.append(env.quadruped.com[0])
    comy.append(env.quadruped.com[1])
    comz.append(env.quadruped.com[2])
#"""

fig1, ax1 = plt.subplots(1,1,figsize = (5,5))
ax1.plot(ALx, label = 'x')
ax1.plot(ALy, label = 'y')
ax1.plot(ALz, label = 'z')
ax1.legend()
ax1.set_ylabel('AL')
ax1.set_xlabel('time')
fig1.savefig('rl/out_dir/ideal_gait_AL.png')

fig2, ax2 = plt.subplots(1,1,figsize = (5,5))
ax2.plot(Ax, label = 'x')
ax2.plot(Ay, label = 'y')
ax2.plot(Az, label = 'z')
ax2.legend()
ax2.set_ylabel('A')
ax2.set_xlabel('time')
fig2.savefig('rl/out_dir/ideal_gait_A.png')

fig3, ax3 = plt.subplots(1,1,figsize = (5,5))
ax3.plot(AFx, label = 'x')
ax3.plot(AFy, label = 'y')
ax3.plot(AFz, label = 'z')
ax3.legend()
ax3.set_ylabel('AF')
ax3.set_xlabel('time')
fig3.savefig('rl/out_dir/ideal_gait_AF.png')

fig4, ax4 = plt.subplots(1,1,figsize = (5,5))
ax4.plot(BLx, label = 'x')
ax4.plot(BLy, label = 'y')
ax4.plot(BLz, label = 'z')
ax4.legend()
ax4.set_ylabel('BL')
ax4.set_xlabel('time')
fig4.savefig('rl/out_dir/ideal_gait_BL.png')

fig5, ax5 = plt.subplots(1,1,figsize = (5,5))
ax5.plot(Bx, label = 'x')
ax5.plot(By, label = 'y')
ax5.plot(Bz, label = 'z')
ax5.legend()
ax5.set_ylabel('B')
ax5.set_xlabel('time')
fig5.savefig('rl/out_dir/ideal_gait_B.png')

fig6, ax6 = plt.subplots(1,1,figsize = (5,5))
ax6.plot(BFx, label = 'x')
ax6.plot(BFy, label = 'y')
ax6.plot(BFz, label = 'z')
ax6.legend()
ax6.set_ylabel('BF')
ax6.set_xlabel('time')
fig6.savefig('rl/out_dir/ideal_gait_BF.png')

fig7, ax7 = plt.subplots(1,1,figsize = (5,5))
ax7.plot(comx, label = 'x')
ax7.plot(comy, label = 'y')
ax7.plot(comz, label = 'z')
ax7.legend()
ax7.set_ylabel('com')
ax7.set_xlabel('time')
fig7.savefig('rl/out_dir/ideal_gait_com.png')

fig8, ax8 = plt.subplots(1,1,figsize = (5,5))
ax8.plot(rewards)
ax8.set_ylabel('reward')
ax8.set_xlabel('time')
fig8.savefig('rl/out_dir/ideal_gait_rewards.png')

fig9, ax9 = plt.subplots(1,1,figsize = (5,5))
ax9.plot(tb)
ax9.set_ylabel('Tb')
ax9.set_xlabel('time')
fig9.savefig('rl/out_dir/ideal_gait_tb.png')

fig10, ax10 = plt.subplots(1,1,figsize = (5,5))
ax10.plot(d1)
ax10.set_ylabel('d1')
ax10.set_xlabel('time')
fig10.savefig('rl/out_dir/ideal_gait_d1.png')

fig11, ax11 = plt.subplots(1,1,figsize = (5,5))
ax11.plot(mu)
ax11.set_ylabel('mu')
ax11.set_xlabel('time')
fig11.savefig('rl/out_dir/ideal_gait_mu.png')

fig12, ax12 = plt.subplots(1,1,figsize = (5,5))
ax12.plot(d3)
ax12.set_ylabel('d3')
ax12.set_xlabel('time')
fig12.savefig('rl/out_dir/ideal_gait_d3.png')

fig13, ax13 = plt.subplots(1,1,figsize = (5,5))
ax13.plot(rewards)
ax13.set_ylabel('rewards')
ax13.set_xlabel('time')
fig13.savefig('rl/out_dir/ideal_gait_rewards.png')

fig14, ax14 = plt.subplots(1,1,figsize = (5,5))
ax14.plot(rewards)
ax14.set_ylabel('rewards')
ax14.set_xlabel('time')
fig14.savefig('rl/out_dir/ideal_gait_rewards.png')

fig15, ax15 = plt.subplots(1,1,figsize = (5,5))
ax15.plot(rewards)
ax15.set_ylabel('rewards')
ax15.set_xlabel('time')
fig15.savefig('rl/out_dir/ideal_gait_rewards.png')

fig16, ax16 = plt.subplots(1,1,figsize = (5,5))
ax16.plot(d2)
ax16.set_ylabel('d2')
ax16.set_xlabel('time')
fig16.savefig('rl/out_dir/ideal_gait_d2.png')


pkl = open('rl/out_dir/ideal_gait_rewards.pickle', 'wb')
pickle.dump(rewards, pkl)
pkl.close()

pkl = open('rl/out_dir/ideal_gait_deltas.pickle', 'wb')
pickle.dump(deltas, pkl)
pkl.close()

pkl = open('rl/out_dir/ideal_gait_tb.pickle', 'wb')
pickle.dump(tb, pkl)
pkl.close()

pkl = open('rl/out_dir/d1.pickle', 'wb')
pickle.dump(d1, pkl)
pkl.close()

pkl = open('rl/out_dir/d2.pickle', 'wb')
pickle.dump(d2, pkl)
pkl.close()

pkl = open('rl/out_dir/d3.pickle', 'wb')
pickle.dump(d3, pkl)
pkl.close()

pkl = open('rl/out_dir/mu.pickle', 'wb')
pickle.dump(mu, pkl)
pkl.close()
