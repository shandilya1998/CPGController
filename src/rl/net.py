import tensorflow as tf
from layers import actor, critic

class ActorNetwork(object):
    def __init__(self, params):
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.TAU = params['TAU']
        self.LEARNING_RATE = params['LRA']

        #Now create the model
        self.model , self.weights, self.state = \
            self.create_actor_network(params)
        self.target_model, self.target_weights, self.target_state = \
            self.create_actor_network(params)

    def train(self, states, action_grads):
        return None

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + \
                (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, params):
        print('[DDPG] Building the actor model')
        S = [
            tf.keras.Input(
                shape = spec.shape,
                dtype = spec.dtype
            ) for spec in params['observation_spec']
        ]

        outputs = actor.get_actor(params)(S)
        model = tf.keras.Model(inputs = S, outputs = outputs)
        return model, model.trainable_weights, model.inputs

class CriticNetwork(object):
    def __init__(self, params):
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.TAU = params['TAU']
        self.LEARNING_RATE = params['LRC']
        self.action_size = params['action_dim']

        # Now create the model
        self.model, self.action, self.state = \
            self.create_critic_network(params)
        self.target_model, self.target_action, self.target_state = \
            self.create_critic_network(params)

    def gradients(self, states, actions):
        return None

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + \
                (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, params):
        log('[DDPG] Building the critic model')
        cr = critic.get_critic(params)

        S = [ 
            tf.keras.Input(
                spec.shape, 
                spec.dtype
            ) for spec in params['observation_spec']
        ]

        A = [
            tf.keras.Input(
                spec.shape,
                spec.dtype
            ) for spec in params['action_spec']
        ]

        out = cr([S, A])

        model = tf.keras.Model(
            inputs = [S, A],
            outputs = [out]
        )

        return model, A, S
