import tensorflow as tf
import tf_agents as tfa

class Env(tfa.environments.tf_environment.TFEnvironment):
    def __init__(self, time_step_spec, action_spec, batch_size):
        super(Env, self).__init__(time_step_spec, action_spec, batch_size)

    def step(self, action):
        
