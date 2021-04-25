import tensorflow as tf
from layers import actor, critic, oscillator, complex
import copy
import os
import numpy as np
from tqdm import tqdm
import time

def swap_batch_timestep(input_t):
    # Swap the batch and timestep dim for the incoming tensor.
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return tf.transpose(input_t, axes)

class TimeDistributed(tf.keras.Model):
    def __init__(self, params, rnn_cell, name, \
            state_sizes, steps = None, return_state = False, trainable = True):
        super(TimeDistributed, self).__init__(
            name = name,
            trainable = trainable
        )
        self.rnn_cell = rnn_cell
        self.num_outputs = len(self.rnn_cell.outputs)
        self.state_sizes = state_sizes
        self.num_states = len(self.state_sizes)
        if steps is None:
            self.steps = copy.deepcopy(params['max_steps'])
        else:
            self.steps = steps
        self.return_state = return_state

    def get_config(self):
        return self.rnn_cell.get_config()

    def call(self, inputs):
        states = inputs[-1 * self.num_states:]
        inputs = inputs[:-1 * self.num_states]
        arrays = [
            tf.TensorArray(
                dtype = tf.dtypes.float32,
                size = 0,
                dynamic_size = True
            ) for i in range(len(inputs))
        ]

        inputs = [swap_batch_timestep(inp) for inp in inputs]
        steps = [inp.shape[0] for inp in inputs]
        batch_size = [inp.shape[1] for inp in inputs]
        batch_size = list(set(batch_size))
        steps = list(set(steps))
        if len(steps) > 1:
            raise ValueError('All input Tensors must be of the same length')
        if len(batch_size) > 1:
            raise ValueError('All input Tensors must be of the same length')
        steps = steps[0]
        batch_size = batch_size[0]
        if steps != self.steps:
            self.steps = steps
        arrays = [
            array.unstack(inp) for array, inp in zip(arrays, inputs)
        ]
        outputs = [
            tf.TensorArray(
                size = 0,
                dtype = tf.dtypes.float32,
                dynamic_size = True
            ) for i in range(self.num_outputs)
        ]
        step = tf.constant(0, dtype = tf.dtypes.int32)
        def cond(step, outputs, states):
            return tf.math.less(step, self.steps)

        def body(step, outputs, states):
            rnn_inp = [inp.read(step) for inp in arrays] + states
            rnn_out = self.rnn_cell(rnn_inp)
            states = rnn_out[-1 * self.num_states:]
            outputs = [
                out.write(step,rnn_o) for out, rnn_o in zip(outputs,rnn_out)
            ]
            step = tf.math.add(step, tf.constant(1, dtype=tf.dtypes.int32))
            return step, outputs, states

        step, outputs, states = tf.while_loop(
            cond, body, [step, outputs, states]
        )

        outputs = [
            swap_batch_timestep(out.stack()) for out in outputs
        ]
        for i, out in enumerate(outputs):
            shape = outputs[i].shape.as_list()
            if len(shape) > 2:
                shape[1] = self.steps
                outputs[i] = tf.ensure_shape(
                    outputs[i], shape
                )
        if self.return_state:
            return outputs, states
        else:
            return outputs

class ActorNetwork(object):
    def __init__(self, params, create_target = True):
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.TAU = params['TAU']
        self.LEARNING_RATE = params['LRA']
        self.params = copy.deepcopy(params)
        #Now create the model
        """
        self.model , self.weights, self.state = \
            self.create_actor_network(self.params, \
            steps = self.params['max_steps'])
        if create_target:
            self.target_model, self.target_weights, self.target_state = \
                self.create_actor_network(self.params, \
                steps = self.params['max_steps'] + 1)
        """
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate = self.LEARNING_RATE
        )
        self.pretrain_loss = tf.keras.losses.MeanSquaredError()
        self.recurrent_state_init = []
        self.recurrent_state_init.append(
            tf.zeros(
                shape = (1, params['units_combine_rddpg'][0]),
                dtype = tf.dtypes.float32
            )
        )
        self.recurrent_state_init.append(
            tf.zeros(
                shape = (1, params['units_omega'][0]),
                dtype = tf.dtypes.float32
            )
        )

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

    def _pretrain_loss(self, y_true, y_pred):
        return self.pretrain_loss(y_true, y_pred)

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + \
                (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_param_net(self, params, trainable = True):
        desired_motion = tf.keras.Input(
            shape = params['observation_spec'][0].shape,
            dtype = params['observation_spec'][0].dtype,
            name = params['observation_spec'][0].name + '_omega_net'
        )

        motion_encoder = tf.keras.Sequential(name = 'motion_encoder_omega_net')
        for i, units in enumerate(params['units_motion_state']):
            motion_encoder.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = 'elu',
                    name = 'motion_state_dense_{i}_omega_net'.format(i = i),
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3),
                    trainable = trainable
                )
            )
        omega_dense = tf.keras.Sequential(name = 'omega_dense_omega_net')
        for i, units in enumerate(params['units_omega']):
            omega_dense.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = 'elu',
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3),
                    name = 'omega_dense_{i}_omega_net'.format(i = i),
                    trainable = trainable
                )
            )
        omega_dense.add(
            tf.keras.layers.Dense(
                units = 1,
                activation = 'relu',
                name = 'omega_dense_out_omega_net',
                kernel_regularizer = tf.keras.regularizers.l2(1e-3),
                trainable = trainable
            )
        )

        mu_dense = tf.keras.Sequential(name = 'mu_dense_omega_net')
        for i, units in enumerate(params['units_mu']):
            mu_dense.add(
                tf.keras.layers.Dense(
                    units = units,
                    activation = 'elu',
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3),
                    name = 'mu_dense_{i}_omega_net'.format(i = i),
                    trainable = trainable
                )
            )
        mu_dense.add(
            tf.keras.layers.Dense(
                units = params['units_osc'],
                activation = 'relu',
                name = 'mu_dense_out_omega_net',
                kernel_regularizer = tf.keras.regularizers.l2(1e-3),
                trainable = trainable
            )
        )

        x = motion_encoder(desired_motion)
        omega = omega_dense(x)
        mu = mu_dense(x)
        model = tf.keras.Model(
            inputs = [desired_motion],
            outputs = [omega, mu],
            name = 'param_net'
        )
        return model

    def create_rhythm_generator(self, params, trainable = True):
        Z = tf.keras.Input(
            shape = (2 * params['units_osc'],),
            dtype = tf.dtypes.float32,
            name = 'Z_rhythm_generator'
        )
        omega = tf.keras.Input(
            shape = (1,),
            dtype = tf.dtypes.float32,
            name = 'omega_rhythm_generator'
        )
        mu = tf.keras.Input(
            shape = (params['units_osc'],),
            dtype = tf.dtypes.float32,
            name = 'mu_rhythm_generator'
        )
        mod_state = tf.keras.Input(
            shape = (2 * params['units_osc'],),
            dtype = tf.dtypes.float32,
            name = 'mod_state_rhythm_generator'
        )

        hopf = oscillator.HopfOscillator(
            units = params['units_osc'],
            dt = params['dt'],
            name = 'hopf_oscillator_rhythm_generator',
            dtype = tf.dtypes.float32,
            trainable = trainable
        )

        add = tf.keras.layers.Add(
            name = 'combine_rhythm_generator',
            trainable = trainable
        )

        complex_mlp = tf.keras.Sequential(
            name = 'complex_mlp_rhythm_generator'
        )
        for i, units in enumerate(params['units_output_mlp'][:-1]):
            complex_mlp.add(
                complex.ComplexDense(
                    units = units,
                    activation = 'elu',
                    name = 'complex_mlp_{i}_rhythm_generator'.format(i = i),
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3),
                    trainable = trainable
                )
            )
        complex_mlp.add(
            complex.ComplexDense(
                units = params['units_output_mlp'][-1],
                activation = 'tanh',
                name = 'complex_mlp_out_rhythm_generator'.format(i = i),
                kernel_regularizer = tf.keras.regularizers.l2(1e-3),
                trainable = trainable
            )
        )

        z = hopf([Z, omega, mu])
        state = add([z, mod_state])
        action = complex_mlp(state)
        action_real, action_imag = tf.split(
            action,
            2,
            axis = -1,
            name = 'get_real_rhythms_rhythm_generator'
        )

        action_real = 2 * action_real

        model = tf.keras.Model(
            inputs = [mod_state, Z, omega, mu],
            outputs = [action_real, z, omega, mu],
            name = 'rhythm_generator'
        )
        return model

    def create_pretrain_actor_cell(self, params, trainable = True):
        desired_motion = tf.keras.Input(
            shape = (
                params['observation_spec'][0].shape[-1]
            ),
            dtype = params['observation_spec'][0].dtype,
            name = params['observation_spec'][0].name + '_cell'
        )
        mod_state = tf.keras.Input(
            shape = (
                2 * params['units_osc']
            ),
            dtype = tf.dtypes.float32,
            name = 'mod_state' + '_cell'
        )
        z = tf.keras.Input(
            shape = (2 * params['units_osc'],),
            dtype = tf.dtypes.float32,
            name = 'z' + '_cell'
        )

        param_net = self.create_param_net(
            params,
            trainable
        )

        rhythm_generator = self.create_rhythm_generator(
            params,
            trainable
        )

        omega, mu = param_net(desired_motion)
        actions, Z, _, _ = rhythm_generator([
            mod_state, z, omega, mu
        ])

        model = tf.keras.Model(
            inputs = [desired_motion, mod_state, z],
            outputs = [actions, omega, mu, Z],
            name = 'pretrain_actor_cell'
        )
        return model

    def create_data(self, path, signal_gen, env):
        env.quadruped.reset()
        F = []
        Y = []
        Z = []
        MU = []
        X = [[] for j in range(len(self.params['observation_spec']))]
        count = 0
        for y, x, f_ in tqdm(signal_gen.generator()):
            f_ = f_ * 2 * np.pi
            y = y * np.pi / 180.0
            env.quadruped.set_motion_state(x)
            _state = env.quadruped.get_state_tensor()
            for i in range(self.params['max_steps']):
                for j in range(self.params['rnn_steps']):
                    ac = y[
                        i * self.params[
                            'rnn_steps'
                        ] + j
                    ]
                    for k, s in enumerate(_state):
                        X[k].append(s)
                    MU.append(
                        np.ones((
                            1, self.params['units_osc']
                        ))
                    )
                    Y.append(np.expand_dims(ac, 0))
                    F.append(np.array([[f_]], dtype = np.float32))
                    if np.isinf(ac).any():
                        print('Inf in unprocessed')
                        continue
                    env.quadruped.all_legs.move(ac)
                    ac = ac / (np.pi / 3)
                    env.quadruped._hopf_oscillator(
                        f_,
                        np.ones((self.params['units_osc'],)),
                        np.zeros((self.params['units_osc'],)),
                    )
                    _state = env.quadruped.get_state_tensor()
                    Z.append(np.expand_dims(_state[-1], 0))
            env.quadruped.reset()
            count += 1
        for j in range(len(X)):
            X[j] = np.concatenate(X[j], axis = 0)
        Y = np.concatenate(Y, axis = 0)
        MU = np.concatenate(MU, axis = 0)
        F = np.concatenate(F, axis = 0)
        Z = np.concatenate(Z, axis = 0)
        num_data = Y.shape[0]
        print('[Actor] Y Shape : {sh}'.format(sh=Y.shape))
        print('[Actor] X Shapes:')
        for i in range(len(X)):
            print('[Actor] {sh}'.format(sh = X[i].shape))
        print('[Actor] F Shape : {sh}'.format(sh=F.shape))
        print('[Axtor] Z Shape : {sh}'.format(sh=Z.shape))
        np.save(os.path.join(path, 'Y.npy'), \
            Y, allow_pickle = True, fix_imports=True)
        np.save(os.path.join(path, 'Z.npy'), \
            Z, allow_pickle = True, fix_imports=True)
        np.save(os.path.join(path, 'MU.npy'), \
            MU, allow_pickle = True, fix_imports=True)
        np.save(os.path.join(path, 'F.npy'), \
            F, allow_pickle = True, fix_imports=True)
        for j in range(len(X)):
            np.save(os.path.join(path, 'X_{j}.npy'.format(j=j)), \
                X[j], allow_pickle = True, fix_imports=True)

    def create_pretrain_dataset(self, data_dir, params, train = True, output_dir = None):
        if output_dir is None:
            output_dir = data_dir
        Y = np.load(
            os.path.join(data_dir, 'Y.npy'),
            allow_pickle = True,
            fix_imports=True
        )
        indices = np.random.choice(Y.shape[0], params['num_data'], replace = False)
        np.save(os.path.join(output_dir, 'indices.npy'), \
            indices, allow_pickle = True, fix_imports=True
        )
        Y = Y[indices]
        Z = np.load(
            os.path.join(data_dir, 'Z.npy'),
            allow_pickle = True,
            fix_imports=True
        )[indices]
        MU = np.load(
            os.path.join(data_dir, 'MU.npy'),
            allow_pickle = True,
            fix_imports=True
        )[indices]
        num_data = Y.shape[0]
        steps = Y.shape[1]
        #Y = Y * (np.pi / 3)
        desired_motion = np.load(
            os.path.join(data_dir, 'X_0.npy'),
            allow_pickle = True,
            fix_imports=True
        )[indices]
        mod_state = np.zeros(
            shape = (
                num_data,
                2 * params['units_osc']
            )
        )
        z =  np.load(
            os.path.join(data_dir, 'X_2.npy'),
            allow_pickle = True,
            fix_imports=True
        )[indices]
        F = np.load(
            os.path.join(data_dir, 'F.npy'),
            allow_pickle = True,
            fix_imports=True
        )[indices]
        Y = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(Y)
        )
        Z = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(Z)
        )
        MU = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(MU)
        )
        F = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(F)
        )
        desired_motion = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(desired_motion)
        )
        mod_state = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(mod_state)
        )
        z = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(z)
        )
        X = tf.data.Dataset.zip((
            desired_motion,
            mod_state,
            z
        ))
        Y = tf.data.Dataset.zip((
            Y, F, MU, Z
        ))
        dataset = tf.data.Dataset.zip((X, Y))
        dataset = dataset.shuffle(num_data // 10)
        num_data = int(num_data * params['train_test_split'])
        print('[Actor] Dataset {ds}'.format(ds = dataset))
        train_dataset = dataset.take(num_data).batch(
            params['pretrain_bs'],
            drop_remainder=True
        ).prefetch(tf.data.AUTOTUNE)
        test_dataset = dataset.skip(num_data).batch(
            params['pretrain_bs'],
            drop_remainder=True
        ).prefetch(tf.data.AUTOTUNE)
        return train_dataset, test_dataset

    def make_untrainable(self, model):
        for layer in model.layers:
            layer.trainable = False
        return model

    def set_model(self, model):
        self.model = model

class CriticNetwork(object):
    def __init__(self, params):
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.TAU = params['TAU']
        self.LEARNING_RATE = params['LRC']
        self.action_size = params['action_dim']
        self.params = copy.deepcopy(params)
        # Now create the model
        self.model, self.action, self.state = \
            self.create_critic_network(self.params, \
            steps = self.params['max_steps'])
        self.target_model, self.target_action, self.target_state = \
            self.create_critic_network(self.params, \
            steps = self.params['max_steps'] + 1)
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate = self.LEARNING_RATE
        )
        self.recurrent_state_init = [
            tf.zeros(
                shape = (1, params['units_gru_rddpg']),
                dtype = tf.dtypes.float32
            )
        ]
        self.mse =tf.keras.losses.MeanSquaredError()

    def q_grads(self, states, actions, a, b, rc_state):
        with tf.GradientTape() as tape:
            watch = actions + [a, b]
            tape.watch(watch)
            inputs = states + actions + [a, b] + rc_state
            q_values, _ = self.model(inputs)
        return tape.gradient(q_values, watch)

    def train(self, states, actions, a, b, rc_state, \
            y, per = False, W = None):
        with tf.GradientTape() as tape:
            inputs = states + actions + [a, b, rc_state]
            y_pred, _ = self.model(inputs)
            deltas = None
            loss = self.loss(y, y_pred, sample_weight = W)
            if per:
                deltas = y_pred - y
        critic_grads = tape.gradient(
            loss,
            self.model.trainable_variables
        )
        self.optimizer.apply_gradients(zip(
            critic_grads,
            self.model.trainable_variables
        ))
        if per:
            return loss, deltas
        else:
            return loss

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + \
                (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def loss(self, y_true, y_pred, sample_weight):
        return self.mse(y_true, y_pred, sample_weight)

    def create_critic_cell(self, params):
        print('[DDPG] Building the critic cell')

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

        a = tf.keras.Input(
            shape = (params['action_dim'],),
            dtype = params['action_spec'][0].dtype
        )

        b = tf.keras.Input(
            shape = (params['action_dim'],),
            dtype = params['action_spec'][0].dtype
        )

        state = tf.keras.Input(
            shape = (params['units_gru_rddpg'],),
            dtype = tf.dtypes.float32
        )

        self.recurrent_state_init = [
            tf.zeros(
                shape = (1, params['units_gru_rddpg']),
                dtype = tf.dtypes.float32
            )
        ]

        inputs = S + A + [a, b, state]

        cr = get_critic_v2(params)
        out = cr(inputs)
        model = tf.keras.Model(
            inputs = inputs,
            outputs = out
        )

        return model, A, S

    def create_critic_network(self, params, steps = None):
        if steps is not None:
            params['max_steps'] = steps
        critic_cell, _, _ = self.create_critic_cell(params)
        print('[DDPG] Building the critic model')
        S = [
            tf.keras.Input(
                shape = (params['max_steps'], spec.shape[-1]),
                dtype = spec.dtype
            ) for spec in params['observation_spec']
        ]

        A = [
            tf.keras.Input(
                shape = (
                    params['max_steps'],
                    spec.shape[-2],
                    spec.shape[-1]),
                dtype = spec.dtype
            ) for spec in params['action_spec']
        ]

        a = tf.keras.Input(
            shape = (params['max_steps'], params['action_dim']),
            dtype = params['action_spec'][0].dtype
        )

        b = tf.keras.Input(
            shape = (params['max_steps'], params['action_dim']),
            dtype = params['action_spec'][0].dtype
        )

        state = tf.keras.Input(
            shape = (params['units_gru_rddpg'],),
            dtype = tf.dtypes.float32
        )

        inputs = S + A + [a, b, state]

        outputs = TimeDistributed(
            params,
            critic_cell,
            'TimeDistributedCritic',
            [
                params['units_gru_rddpg']
            ],
            trainable = True
        )(inputs)
        model = tf.keras.Model(
            inputs = inputs,
            outputs = outputs
        )
        return model, A, S
