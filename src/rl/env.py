import tensorflow as tf
import tf_agents as tfa
from rl.constants import params
from simulations.ws.src.quadruped.scripts.quadruped import Quadruped
from tf_agents.trajectories.time_step import TimeStep, time_step_spec

def swap_batch_timestep(input_t):
    # Swap the batch and timestep dim for the incoming tensor.
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return tf.transpose(input_t, axes)

class Env(tfa.environments.tf_environment.TFEnvironment):
    def __init__(self,
            time_step_spec,
            params,
            experiment,
            initial_state = None,
            GUI = False
        ):
        super(Env, self).__init__(time_step_spec, params['action_spec'], 1)
        if initial_state is None:
            initial_state = [
                tf.expand_dims(tf.zeros(
                    spec.shape, 
                    spec.dtype
                ), 0) for spec in params['observation_spec']
            ]

        self._action_init = [
            tf.expand_dims(tf.zeros(
                spec.shape,
                spec.dtype
            ), 0) for spec in self.action_spec()
        ]
        self._action = self._action_init
        self.params = params
        self.initial_state = initial_state
        self._episode_ended = False
        self._state = self.initial_state
        self._reward = 0.0
        self.current_time_step = self._create_initial_time_step()
        self.quadruped = Quadruped(params, experiment)
        self.params = params

    def _create_initial_time_step(self):
        discount = tf.ones((1,), dtype = tf.dtypes.float32)
        step_type = tf.stack([tfa.trajectories.time_step.StepType.FIRST \
                        for i in range(self.batch_size)], 0)
        return TimeStep(step_type, self._reward, discount, self._state)

    def _reset(self):
        print('[DDPG] Resetting Environment')
        self._episode_ended = True
        self._state, self._reward = self.quadruped.reset()
        self._state = [tf.expand_dims(
            tf.convert_to_tensor(
                state
            ),
            0
        ) for state in self._state]
        self._reward = tf.expand_dims(
            tf.convert_to_tensor(
                self._reward
            ),
            0
        )
        self.current_time_step = self._create_initial_time_step()
        return self.current_time_step

    def step(self, action, desired_motion, last_step = False):
        return self._step(action, desired_motion, last_step)

    def set_motion_state(self, desired_motion):
        self.quadruped.set_motion_state(desired_motion)

    def set_osc_state(self, osc_state):
        self.quadruped.set_osc_state = osc_state

    def _step(self, action, desired_motion, last_step = False):
        observation = self.quadruped.get_state()
        reward = 0.0
        action[0] = swap_batch_timestep(action[0])
        action[1] = swap_batch_timestep(action[1])
        self.quadruped.set_last_pos()
        for i in range(self.params['rnn_steps']):
            _action = [action[0][i], action[1][i]]
            observation =  self.quadruped.step(
                _action,
                desired_motion
            )
            reward += 1e-3 * self.quadruped.get_COT()
            reward += self.quadruped.get_motion_reward()
        self.quadruped.set_support_lines()
        reward += 1e-5 * self.quadruped.get_stability_reward()
        #reward += self.quadruped.reward
        action[0] = swap_batch_timestep(action[0])
        action[1] = swap_batch_timestep(action[1])
        observation = [
            tf.expand_dims(
                tf.convert_to_tensor(ob),
                0
            ) for ob in observation
        ]
        reward = tf.convert_to_tensor(reward, dtype = tf.dtypes.float32)
        step_type = tfa.trajectories.time_step.StepType.MID
        self._episode_ended = last_step
        if last_step:
            step_type = tfa.trajectories.time_step.StepType.LAST
        if not self.quadruped.upright:
            step_type = tfa.trajectories.time_step.StepType.LAST
        step_type = tf.stack([step_type \
            for i in range(self.batch_size)])

        discount = tf.ones((self.batch_size,), dtype = tf.dtypes.float32)

        self.current_time_step = TimeStep(
            step_type,
            reward,
            discount,
            observation,
        )

        return self.current_time_step

    def _current_time_step(self):
        return self.current_time_step
