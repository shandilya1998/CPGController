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
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate = self.LEARNING_RATE
        )

    def train(self, states, q_grads):
        with tf.GradientTape() as tape:
            out = self.model(states)
        grads = tape.gradient(
            out,
            self.model.trainable_variables,
            [-grad for grad in q_grads]
        )
        self.optimizer.apply_gradients(
            zip(
                grads,
                self.model.trainable_variables
            )
        )

    def _pretrain_loss(self, y_true, y_pred):
        return tf.math.abs(
            tf.math.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred))
        )

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
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
        self.model, self.action, self.state, self.history = \
            self.create_critic_network(params)
        self.target_model, self.target_action, self.target_state, \
            self.target_history = self.create_critic_network(params)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate = self.LEARNING_RATE
        )


    def q_grads(self, states, actions):
        with tf.GradientTape() as tape:
            tape.watch(actions)
            inputs = states + actions
            q_values = self.model(inputs)
            q_values = tf.squeeze(q_values)
        return tape.gradient(q_values, actions)

    def train(self, states, actions, y):
        with tf.GradientTape() as tape:
            inputs = states + actions
            y_pred = self.model(inputs)
            loss = self.loss(y, y_pred)
        critic_grads = tape.gradient(
            loss,
            self.model.trainable_variables
        )
        self.optimizer.apply_gradients(zip(
            critic_grads,
            self.model.trainable_variables
        ))
        return loss

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + \
                (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def loss(self, y_true, y_pred):
        return tf.math.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred))

    def create_critic_network(self, params):
        print('[DDPG] Building the critic model')

        S = [
            tf.keras.Input(
                shape = spec.shape, 
                dtype = spec.dtype
            ) for spec in params['observation_spec']
        ]

        A = [
            tf.keras.Input(
                shape = spec.shape,
                dtype = spec.dtype
            ) for spec in params['action_spec']
        ]

        history = tf.keras.Input(
            shape = params['history_spec'].shape,
            dtype = params['history_spec'].dtype
        )

        cr = critic.get_critic(params)
        out = cr([S, A, history])

        model = tf.keras.Model(
            inputs = [S, A, history],
            outputs = out
        )

        return model, A, S, history
