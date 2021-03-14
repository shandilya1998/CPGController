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
signal_gen = Signal(2*params['rnn_steps'] + 1, dt)
signal_gen.build(10, 30, 45, 30 )
y, _ = signal_gen.get_signal()
y = y[:, 1:] * np.pi/180
v = signal_gen.compute_v((0.1+0.015)*2.2)
desired_motion = np.array([1, 0, 0, v, 0 ,0], dtype = np.float32)
mu = np.array([30, 30 / 5, 45])
mu = [mu for i in range(4)]
f = 2 * np.pi * frequency_estimator.freq_from_autocorr(y[:, 0], dt)

fig, axes = plt.subplots(4, 1, figsize = (20,20))
legs = ['Front Right', 'Front Left', 'Back Right', 'Back Left']

t = np.arange(params['rnn_steps']) * dt
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
for i in range(params['rnn_steps']):
    signal = y[:, i : i + params['rnn_steps'], :]
    print(signal.shape)
    current_time_step=env.step([signal,osc],desired_motion)
#"""
