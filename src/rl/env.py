import tensorflow as tf
import tf_agents as tfa
from src.rl.constants import *
from src.simulations.ws.src.quadruped_control.robot import Quadruped

class Env(tfa.environments.tf_environment.TFEnvironment):
    def __init__(self, 
            time_step_spec, 
            action_spec, 
            batch_size,
            initial_state = None):
        super(Env, self).__init__(time_step_spec, action_spec, batch_size)
       
        if initial_state is None:
            initial_state = [
                tf.zeros(
                    spec.shape, 
                    spec.dtype
                ) for spec in observation_spec
            ]
        
        self.initial_state = initial_state
        self.reward_spec = self.time_step_spec.reward_spec
        self._episode_ended = False
        self._state = self.initial_state
        self.current_time_step = self._create_initial_time_step()
        self.quadruped = Quadruped()

    def _create_initial_time_step(self):
        reward = tf.nest.map_structure(
            lambda r: tf.zeros([self.batch_size] + list(r.shape), dtype=r.dtype),
            self.reward_spec
        )
        discount = tf.ones((self.batch_size,), dtype = tf.dtypes.float32)
        step_type = tf.stack([tfa.trajectories.StepType.FIRST for i in range self.batch_size])
        return tfa.trajectories.TimeStep(step_type, reward, discount, self._state) 

    def reset(self):
        self._state = self.initial_state
        self._episode_ended = True
        self.current_time_step = self._create_initial_time_step()
        return self.current_time_step
     
    def step(self, action, last_episode = False):
        if self._episode_ended:
            return self.reset()
        else:
            observation, reward = self.quadruped.step(action)
            observation = [
                tf.convert_to_tensor(ob) for ob in observation
            ] + [action[-1]]

            step_type = tfa.trajectories.time_step.StepType.MID
            if last_episode:
                step_type = tfa.trajectories.time_step.StepType.LAST        
            step_type = tf.stack([step_type for i in range self.batch_size])

            discount = tf.ones((self.batch_size,), dtype = tf.dtypes.float32)

            self.current_time_step = tfa.trajectories.time_step.TimeStep(
                step_type.
                reward,
                discount, 
                observation
            )

            return self.current_time_step

        


if __name__ == '__main__':
    time_step_spec = tfa.trajectories.time_step.time_step_spec(
        observation_spec,
        reward_spec
    ) 
    
    initial_state = [tf.zeros(spec.shape, spec.dtype) for spec in observation_spec]
    env = Env(
        time_step_spec,
        action_spec,
        params['BATCH_SIZE'],
        initial_state
    )
