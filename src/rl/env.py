import tensorflow as tf
import tf_agents as tfa
from rl.constants import params
from simulations.ws.src.quadruped.scripts import quadruped
from tf_agents.trajectories.time_step import TimeStep, time_step_spec

class Env(tfa.environments.tf_environment.TFEnvironment):
    def __init__(self,
            time_step_spec,
            params,
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
        self.reward_spec = self.time_step_spec.reward_spec
        self._episode_ended = False
        self._state = self.initial_state
        self._reward = 0.0
        self.current_time_step = self._create_initial_time_step()
        self.quadruped = robot.Quadruped(params)
        self.params = params

    def _create_initial_time_step(self):
        discount = tf.ones((1,), dtype = tf.dtypes.float32)
        step_type = tf.stack([tfa.trajectories.StepType.FIRST \
                        for i in range(self.batch_size)], 0)
        return TimeStep(step_type, self._reward, discount, self._state)

    def reset(self):
        self._episode_ended = True
        self._state, self._reward = self.quadruped.reset()
        self._state = tf.expand_dims(
            tf.convert_to_tensor(
                self._state
            ),
            0
        )
        self._reward = tf.expand_dims(
            tf.convert_to_tensor(
                self._reward
            ),
            0
        )
        self.current_time_step = self._create_initial_time_step()
        return self.current_time_step

    def step(self, action, last_step = False):
        if self._episode_ended:
            return self.reset()
        else:
            observation, reward = self.quadruped.step(action)
            observation = [
                tf.expand_dims(
                    tf.convert_to_tensor(ob),
                    0
                ) for ob in observation
            ] + [action[-1]]

            step_type = tfa.trajectories.time_step.StepType.MID
            self._episode_ended = last_step
            if last_step:
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

if __name__ == '__main__':
    time_step_spec = _time_step_spec(
        params['observation_spec'],
        params['reward_spec'],
    )

    initial_state = [tf.zeros(spec.shape, spec.dtype) \
        for spec in params['observation_spec']]

    env = Env(
        time_step_spec,
        params['action_spec'],
        params,
        initial_state,
    )
