import tensorflow as tf
from layers import actor, critic

class ActorNetwork(object):
    def __init__(self, params):
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.TAU = params['TAU']
        self.LEARNING_RATE = params['LRA']
        self.params = params
        #Now create the model
        self.model , self.weights, self.state = \
            self.create_actor_network(params)
        self.target_model, self.target_weights, self.target_state = \
            self.create_actor_network(params)
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate = self.LEARNING_RATE
        )
        self.pretrain_loss = tf.keras.losses.MeanSquaredError()


    def train(self, states, q_grads):
        with tf.GradientTape() as tape:
            action, [omega, mu, mean, state] = self.model(states)
            action[0] = action[0] * tf.repeat(
                tf.expand_dims(mu, 1),
                self.params['rnn_steps'],
                axis = 1
            ) + tf.repeat(
                tf.expand_dims(mean, 1),
                self.params['rnn_steps'],
                axis = 1
            )
        grads = tape.gradient(
            action,
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
        return self.pretrain_loss(y_true, y_pred)

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + \
                (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_encoder(self, params, trainable = True):
        S = [
            tf.keras.Input(
                shape = spec.shape,
                dtype = spec.dtype
            ) for spec in params['observation_spec']
        ]

        encoder = actor.get_state_encoder(params, trainable)
        [state, omega, mu, mean] = encoder(S[:2])
        model = tf.keras.Model(inputs = S, outputs = [state, omega, mu, mean]) 
        return model

    def make_untrainable(self, model):
        for layer in model.layers:
            layer.trainable = False
        return model

    def set_model(self, model):
        self.model = model

    def create_actor_network(self, params, encoder = None):
        print('[DDPG] Building the actor model')
        if encoder is None:
            encoder = self.create_encoder(params)
        
        state, omega, mu, mean = encoder.outputs

        [action, z_out] = actor.get_complex_mlp(params)(
            [encoder.inputs[2], state, omega]
        )
        outputs = [[action, z_out], [omega, mu, mean, state]]
        model = tf.keras.Model(inputs = encoder.inputs, outputs = outputs)
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
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate = self.LEARNING_RATE
        )
        self.mse =tf.keras.losses.MeanSquaredError()

    def q_grads(self, states, actions, history, history_osc):
        with tf.GradientTape() as tape:
            tape.watch(actions)
            inputs = states + actions + [history, history_osc]
            q_values = self.model(inputs)
            q_values = tf.squeeze(q_values)
        return tape.gradient(q_values, actions)

    def train(self, states, actions, history, history_osc, y):
        with tf.GradientTape() as tape:
            inputs = states + actions + [history, history_osc]
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
        return self.mse(y_true, y_pred)

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

        history_osc = tf.keras.Input(
            shape = params['history_osc_spec'].shape,
            dtype = params['history_spec'].dtype
        )

        inputs = S + A + [history, history_osc]

        cr = critic.get_critic(params)
        out = cr(inputs)
        model = tf.keras.Model(
            inputs = [S, A, history, history_osc],
            outputs = out
        )

        return model, A, S, history
