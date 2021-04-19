import numpy as np
import tensorflow as tf
import copy

class ActorNetwork(object):
    def __init__(self, params, create_target = True):
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.TAU = params['TAU']
        self.LEARNING_RATE = params['LRA']
        self.params = copy.deepcopy(params)
        #Now create the model
        self.model , self.weights, self.state = \
            self.create_actor_network(self.params, \
            steps = self.params['max_steps'])
        if create_target:
            self.target_model, self.target_weights, self.target_state = \
                self.create_actor_network(self.params, \
                steps = self.params['max_steps'] + 1)
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate = self.LEARNING_RATE
        )
        self.pretrain_loss = tf.keras.losses.MeanSquaredError()

    def train(self, states, rc_state, q_grads):
        with tf.GradientTape() as tape:
            out, osc, omega, a, b, state, z_out, combine_state, \
                    omega_state = self.model(states[:-1] + rc_state)
            action = [out, osc]
        grads = tape.gradient(
            action + [a, b],
            self.model.trainable_variables,
            [-1 * grad for grad in q_grads]
        )
        self.optimizer.apply_gradients(
            zip(
                grads,
                self.model.trainable_variables
            )
        )

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + \
                (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self):

