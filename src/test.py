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

S = [
    np.expand_dims(env.quadruped.motion_state, 0),
    np.expand_dims(env.quadruped.robot_state, 0),
    np.expand_dims(env.quadruped.osc_state, 0)
]

optimizer = tf.keras.optimizers.Adam(
    learning_rate = params['LRA']
)
signal_gen = SignalDataGen(params)

y, x = next(signal_gen.generator())
with tf.GradientTape() as tape:
    y_pred = actor.model(S)
    loss = tf.keras.losses.mean_squared_error(y, y_pred[0])
grads = tape.gradient(loss, actor.model.trainable_variables)
print(grads)
optimizer.apply_gradients(zip(grads, actor.model.trainable_variables))
