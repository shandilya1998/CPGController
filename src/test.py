from rl.net import ActorNetwork
from rl.constants import params
from rl.env import Env
import tf_agents as tfa
import tensorflow as tf
from _learn import SignalDataGen
import numpy as np

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
