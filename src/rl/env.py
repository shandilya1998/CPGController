import tensorflow as tf
import tf_agents as tfa

class Env(tfa.environments.tf_environment.TFEnvironment):
    def __init__(self,):
        super(Env, self).__init__()
        
