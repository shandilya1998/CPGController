import tensorflow as tf
from src.layers import actor, critic
from constants import params

class ActorNetwork(object):
    def __init__(self, params):
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.TAU = params['TAU']
        self.LEARNING_RATE = params['LEARNING_RATE']
        
        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(params)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(params)
        self.action_gradient = tf.placeholder(tf.float32,[None, params['action_dim']])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(params['LEARNING_RATE']).apply_gradients(grads)

    def train(self, states, action_grads):
        return None

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, params):
        log('[DDPG] Building the actor model')
        ac_cell = actor.get_actor_cell(params)
        inputs = [
            tf.keras.Input(
                spec.shape, 
                spec.dtype
            ) for spec in params['observation_spec']
        ]
        outputs = actor.TimeDistributed(ac_cell, params)(inputs)
        model = tf.keras.Model(inputs = inputs, outputs = outputs)
        return model, model.trainable_weights, model.inputs


class CriticNetwork(object):
    def __init__(self, params):
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.TAU = params['TAU']
        self.LEARNING_RATE = params['LEARNING_RATE']
        self.action_size = params['action_dim']

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(params)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(params)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update

    def gradients(self, states, actions):
        return None    

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, params):
        log('[DDPG] Building the critic model')
        cr_cell = critic.get_critic_cell(params)

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

        l1 = critic.TimeDistributed(cr_cell, params)([S[0], S[1], inp4])
        real_1 = tf.math.real(S[2])
        imag_1 = tf.math.imag(S[2])
        real_2 = tf.math.real(A[1])
        imag_2 = tf.math.imag(A[1])
        real = tf.concat([real_1, real_2], axis = -1)
        imag = tf.concat([imag_1, imag_2], axis = -1)
        real = tf.keras.layers.Dense(
            units = params['units_osc'],
            activation = params['lstm_state_dense_activation']
        )(real)
        imag = tf.keras.layers.Dense(
            units = params['units_osc'],
            activation = params['lstm_state_dense_activation']
        )(imag)
        x = tf.concat([real, imag], axis = -1)
        lstm_state = tf.keras.layers.Dense(
            units = params['lstm_units'],
            activation = params['lstm_state_dense_activation']
        )(x)
        hidden = [lstm_state for i in range(4)]
        out = tf.keras.layers.LSTM(
            units = params['lstm_units']
            return_sequences = True
        )(x, hidden)
        
        return model, inp4, S
