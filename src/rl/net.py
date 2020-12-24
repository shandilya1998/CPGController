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
        actor_cell = actor.get_actor_net(params)
        model = actor.TimeDistributed(actor_cell, params)
        return model, model.trainable_weights, model.inputs[:2]


class CriticNetwork(object):
    def __init__(self, params):
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.TAU = params['TAU']
        self.LEARNING_RATE = params['LEARNING_RATE']
        self.action_size = params['action_dim']

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
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
        S = tf.keras.layers.Input(shape=[state_size])
        A = tf.keras.layers.Input(shape=[action_dim], name='action2')
        w1 = tf.keras.layers.Dense(HIDDEN1_UNITS, activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                   kernel_initializer=tf.keras.initializers(minval=-1.0 / np.sqrt(state_size), maxval=1.0 / np.sqrt(state_size)),
                   bias_initializer=tf.keras.initializers(minval=-1.0 / np.sqrt(state_size), maxval=1.0 / np.sqrt(state_size))
                   )(S)
        a1 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                   kernel_initializer=tf.keras.initializers(minval=-1.0 / np.sqrt(action_dim), maxval=1.0 / np.sqrt(action_dim)),
                   bias_initializer=tf.keras.initializers(minval=-1.0 / np.sqrt(action_dim), maxval=1.0 / np.sqrt(action_dim))
                   )(A)
        h1 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                   kernel_initializer=tf.keras.initializers(minval=-1.0 / np.sqrt(HIDDEN1_UNITS), maxval=1.0 / np.sqrt(HIDDEN1_UNITS)),
                   bias_initializer=tf.keras.initializers(minval=-1.0 / np.sqrt(HIDDEN1_UNITS), maxval=1.0 / np.sqrt(HIDDEN1_UNITS))
                   )(w1)
        h2 = tf.keras.layers.merge([h1, a1], mode='sum')
        h3 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                   kernel_initializer=tf.keras.initializers(minval=-1.0 / np.sqrt(HIDDEN2_UNITS), maxval=1.0 / np.sqrt(HIDDEN2_UNITS)),
                   bias_initializer=tf.keras.initializers(minval=-1.0 / np.sqrt(HIDDEN2_UNITS), maxval=1.0 / np.sqrt(HIDDEN2_UNITS))
                   )(h2)
        V = tf.keras.layers.Dense(action_dim, activation='linear',  # Linear activation function
                  kernel_initializer=tf.keras.initializers(minval=-0.003, maxval=0.003),
                  bias_initializer=tf.keras.initializers(minval=-0.003, maxval=0.003))(h3)
        model = tf.keras.Model(input=[S, A], output=V)
        adam = tf.keras.optimizers.Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S
