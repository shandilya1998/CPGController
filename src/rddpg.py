import numpy as np
import tensorflow as tf
import os
import pickle
import time
import math
from rl.rddpg_net import ActorNetwork, CriticNetwork
from rl.env import Env
from rl.constants import params
from rl.replay_buffer import ReplayBuffer, OU
import argparse
import tf_agents as tfa
import matplotlib.pyplot as plt
import matplotlib
from learn import SignalDataGen
from tqdm import tqdm

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class Learner:
    def __init__(self, params, experiment, create_data = False):
        self.params = params
        self.experiment = experiment
        self.actor = ActorNetwork(self.params)
        self.critic = CriticNetwork(self.params)
        self.replay_buffer = ReplayBuffer(self.params)
        self.time_step_spec = tfa.trajectories.time_step.time_step_spec(
            observation_spec = self.params['observation_spec'],
            reward_spec = self.params['reward_spec']
        )
        self.env = Env(
            self.time_step_spec,
            self.params,
            experiment,
            rddpg = True
        )
        self.current_time_step = None
        self._action = self.env._action_init
        self._noise_init = [
            tf.expand_dims(tf.zeros(
                spec.shape,
                spec.dtype
            ), 0) for spec in self.env.action_spec()
        ]
        self._noise = self._noise_init
        self.OU = OU()
        self.dt = self.params['dt']
        self.pretrain_dataset = self.load_ddpg_dataset()
        self.desired_motion = []
        count = 0
        for i, (x, y) in enumerate(self.pretrain_dataset):
            self.desired_motion.append(np.repeat(
                np.expand_dims(x[0][0], 0),
                self.params['max_steps'] + 1,
                0
            ))
            if count > 10:
                break
            count += 1
        count = 0
        for i, (x, y) in enumerate(self.pretrain_dataset):
            x_ = np.zeros(x[0][0].shape, dtype = np.float32)
            x_[0] = -1 / np.sqrt(2, dtype = np.float32)
            x_[1] = -1 / np.sqrt(2, dtype = np.float32)
            x_[3] = x[0][0][4] / np.sqrt(2, dtype = np.float32)
            x_[4] = x[0][0][4] / np.sqrt(2, dtype = np.float32)
            self.desired_motion.append(np.repeat(
                np.expand_dims(x_, 0),
                self.params['max_steps'] + 1,
                0
            ))
            if count > 10:
                break
            count += 1
        count = 0
        for i, (x, y) in enumerate(self.pretrain_dataset):
            x_ = np.zeros(x[0][0].shape, dtype = np.float32)
            x_[0] = 1 / np.sqrt(2, dtype = np.float32)
            x_[1] = -1 / np.sqrt(2, dtype = np.float32)
            x_[3] = -x[0][0][4] / np.sqrt(2, dtype = np.float32)
            x_[4] = x[0][0][4] / np.sqrt(2, dtype = np.float32)
            self.desired_motion.append(np.repeat(
                np.expand_dims(x_, 0),
                self.params['max_steps'] + 1,
                0
            ))
            if count > 10:
                break
            count += 1
        self._state = [
            self.env.quadruped.motion_state,
            self.env.quadruped.robot_state,
            self.env.quadruped.osc_state
        ]
        matplotlib.use('Agg')
        physical_devices = tf.config.list_physical_devices('GPU')
        np.seterr(all='raise')
        print('[Actor] GPU>>>>>>>>>>>>')
        print('[Actor] {lst}'.format(lst = physical_devices))
        self.p1 = 1.0
        self.epsilon = 1
        if create_data:
            self.signal_gen = SignalDataGen(params)
            self.signal_gen.set_N(
                self.params['rnn_steps'] * (self.params['max_steps'] + 2),
                create_data
            )
            self.create_dataset()
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.01,
            decay_steps=20,
            decay_rate=0.95
        )
        self.pretrain_actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate = lr_schedule
        )
        self.action_mse = tf.keras.losses.MeanSquaredError()
        self.omega_mse = tf.keras.losses.MeanSquaredError()
        self.a_mse = tf.keras.losses.MeanSquaredError()
        self.b_mse = tf.keras.losses.MeanSquaredError()

    def create_dataset_v2(self):
        self.env.quadruped.reset()
        F = []
        A = []
        B = []
        Y = []
        X = [[] for j in range(len(self.params['observation_spec']))]
        for y, x, f_ in tqdm(self.signal_gen.generator()):
            f_ = f_ * 2 * np.pi
            y = y * np.pi / 180.0
            self.env.quadruped.set_motion_state(x)
            _state = self.env.quadruped.get_state_tensor()
            x = [[] for j in range(len(self.params['observation_spec']))]
            f = []
            a = []
            b = []
            x.append(_state[-1])
            x.append(self.actor.recurrent_state_init[0])
            x.append(self.actor.recurrent_state_init[1])
            actions = []
            count = 0
            for i in range(self.params['max_steps']):
                for j in range(self.params['rnn_steps']):
                    y_, b_, a_ = self.signal_gen.preprocess(
                        y[
                            i * self.params[
                                'rnn_steps'
                            ] + j: (i + 1) * self.params[
                            'rnn_steps'
                            ] + j
                        ]
                    )
                    a_ = a_ / (np.pi / 3)
                    for k, s in enumerate(_state):
                        X[k].append(s)
                    Y.append(np.expand_dims(y_, 0))
                    B.append(np.expand_dims(b_, 0))
                    A.append(np.expand_dims(a_, 0))
                    F.append(np.array([[f_]], dtype = np.float32))
                    ac = y[count]
                    count += 1
                    if np.isinf(ac).any():
                        print('Inf in unprocessed')
                        continue
                    self.env.quadruped.all_legs.move(ac)
                    self.env.quadruped._hopf_oscillator(
                        f_,
                        np.ones((self.params['units_osc'],)),
                        np.zeros((self.params['units_osc'],)),
                    )
                    _state = self.env.quadruped.get_state_tensor()
            self.env.quadruped.reset()


    def create_dataset(self):
        self.env.quadruped.reset()
        F = []
        A = []
        B = []
        Y = []
        X = [[] for j in range(len(self.params['observation_spec']) + 2)]
        for y, x, f_ in tqdm(self.signal_gen.generator()):
            f_ = f_ * 2 * np.pi
            y = y * np.pi / 180.0
            self.env.quadruped.set_motion_state(x)
            _state = self.env.quadruped.get_state_tensor()
            x = [[] for j in range(len(self.params['observation_spec']) - 1)]
            f = []
            a = []
            b = []
            x.append(_state[-1])
            x.append(self.actor.recurrent_state_init[0])
            x.append(self.actor.recurrent_state_init[1])
            actions = []
            count = 0
            for i in range(self.params['max_steps']):
                y_, b_, a_ = self.signal_gen.preprocess(
                    y[i*self.params['rnn_steps']:(i+1)*self.params['rnn_steps']]
                )
                a_ = a_ / (np.pi / 3)
                for k, s in enumerate(_state[:-1]):
                    x[k].append(s)
                actions.append(np.expand_dims(y_, 0))
                b.append(np.expand_dims(b_, 0))
                a.append(np.expand_dims(a_, 0))
                f.append(np.array([[f_]], dtype = np.float32))
                for j in range(self.params['rnn_steps']):
                    ac = y[count]
                    count += 1
                    if np.isinf(ac).any():
                        print('Inf in unprocessed')
                        continue
                    self.env.quadruped.all_legs.move(ac)
                    self.env.quadruped._hopf_oscillator(
                        f_,
                        np.ones((self.params['units_osc'],)),
                        np.zeros((self.params['units_osc'],)),
                    )
                _state = self.env.quadruped.get_state_tensor()
            for j in range(len(self.params['observation_spec']) - 1):
                x[j] = np.expand_dims(
                    np.concatenate(x[j], axis = 0), 0
                )
            b = np.expand_dims(np.concatenate(b, axis = 0), 0)
            a = np.expand_dims(np.concatenate(a, axis = 0), 0)
            f = np.expand_dims(np.concatenate(f, axis = 0), 0)
            actions = np.expand_dims(np.concatenate(actions, axis = 0), 0)
            for j in range(len(X)):
                X[j].append(x[j])
            B.append(b)
            A.append(a)
            F.append(f)
            Y.append(actions)
            self.env.quadruped.reset()
            _state = self.env.quadruped.get_state_tensor()
            count += 1
        for j in range(len(X)):
            X[j] = np.concatenate(X[j], axis = 0)
        Y = np.concatenate(Y, axis = 0)
        F = np.concatenate(F, axis = 0)
        A = np.concatenate(A, axis = 0)
        B = np.concatenate(B, axis = 0)
        print('[Actor] Y Shape : {sh}'.format(sh=Y.shape))
        print('[Actor] X Shapes:')
        for i in range(len(X)):
            print('[Actor] {sh}'.format(sh = X[i].shape))
        print('[Actor] A Shape : {sh}'.format(sh=A.shape))
        print('[Actor] B Shape : {sh}'.format(sh=B.shape))
        print('[Actor] F Shape : {sh}'.format(sh=F.shape))
        np.save('data/pretrain_rddpg_2/Y.npy', Y, \
            allow_pickle = True, fix_imports=True)
        time.sleep(3)
        np.save('data/pretrain_rddpg_2/F.npy', F, \
            allow_pickle = True, fix_imports=True)
        time.sleep(3)
        np.save('data/pretrain_rddpg_2/A.npy', A, \
            allow_pickle = True, fix_imports=True)
        time.sleep(3)
        np.save(
            'data/pretrain_rddpg_2/B.npy',
            B,
            allow_pickle = True,
            fix_imports=True
        )
        time.sleep(3)
        for j in range(len(X)):
            time.sleep(3)
            np.save('data/pretrain_rddpg_2/X_{j}.npy'.format(j=j), X[j], \
                allow_pickle = True, fix_imports=True)

    def load_ddpg_dataset(self):
        Y = np.load(
            'data/pretrain/Y.npy',
            allow_pickle = True,
            fix_imports=True
        )[:, :self.params['rnn_steps'], :]
        time.sleep(3)
        num_data = Y.shape[0]
        Y = tf.convert_to_tensor(Y)
        F = tf.convert_to_tensor(np.load('data/pretrain/F.npy', allow_pickle = True, fix_imports=True))
        time.sleep(3)
        MU = tf.convert_to_tensor(np.load('data/pretrain/MU.npy', allow_pickle = True, fix_imports=True))
        MEAN = tf.convert_to_tensor(np.load('data/pretrain/MEAN.npy', allow_pickle = True, fix_imports=True))
        X = []
        for j in range(len(self.params['observation_spec'])):
            X.append(
                tf.data.Dataset.from_tensor_slices(
                    tf.convert_to_tensor(
                        np.load('data/pretrain/X_{j}.npy'.format(j=j), allow_pickle = True, fix_imports=True)
                    )
                )
            )
        if num_data == self.params['num_data']:
            self.num_data = self.params['num_data']
        else:
            self.num_data = num_data
        self.num_batches = num_data//self.params['pretrain_bs']
        Y = tf.data.Dataset.from_tensor_slices(Y)
        F = tf.data.Dataset.from_tensor_slices(F)
        MU = tf.data.Dataset.from_tensor_slices(MU)
        MEAN = tf.data.Dataset.from_tensor_slices(MEAN)
        Y = tf.data.Dataset.zip((Y, F, MU, MEAN))
        X = tf.data.Dataset.zip(tuple(X))
        dataset = tf.data.Dataset.zip((X, Y))
        dataset = dataset.shuffle(self.num_data).batch(
            self.params['pretrain_bs'],
            drop_remainder=True
        ).prefetch(tf.data.AUTOTUNE)
        return dataset

    def load_dataset(self):
        Y = np.load(
            'data/pretrain_rddpg/Y.npy',
            allow_pickle = True,
            fix_imports=True
            )[:, :, :self.params['rnn_steps'], :]
        time.sleep(3)
        num_data = Y.shape[0]
        Y = tf.convert_to_tensor(Y)
        F = tf.convert_to_tensor(np.load('data/pretrain_rddpg/F.npy', allow_pickle = True, fix_imports=True))
        time.sleep(3)
        A = tf.convert_to_tensor(np.load('data/pretrain_rddpg/A.npy', allow_pickle = True, fix_imports=True))
        B = tf.convert_to_tensor(np.load('data/pretrain_rddpg/B.npy', allow_pickle = True, fix_imports=True))
        X = []
        for j in range(len(self.params['observation_spec']) + 2):
            X.append(
                tf.data.Dataset.from_tensor_slices(
                    tf.convert_to_tensor(
                        np.load('data/pretrain_rddpg/X_{j}.npy'.format(j=j), allow_pickle = True, fix_imports=True)
                    )
                )
            )
        if num_data == self.params['num_data']:
            self.num_data = self.params['num_data']
        else:
            self.num_data = num_data
        self.num_batches = num_data//self.params['pretrain_bs']
        Y = tf.data.Dataset.from_tensor_slices(Y)
        F = tf.data.Dataset.from_tensor_slices(F)
        A = tf.data.Dataset.from_tensor_slices(A)
        B = tf.data.Dataset.from_tensor_slices(B)
        Y = tf.data.Dataset.zip((Y, F, A, B))
        X = tf.data.Dataset.zip(tuple(X))
        dataset = tf.data.Dataset.zip((X, Y))
        dataset = dataset.shuffle(self.num_data).batch(
            self.params['pretrain_bs'],
            drop_remainder=True
        ).prefetch(tf.data.AUTOTUNE)
        return dataset

    def load_actor(self, path, path_target):
        print('[DDPG] Loading Actor Weights')
        self.actor.model.load_weights(path)
        self.actor.target_model.load_weights(path_target)

    def load_critic(self, path, path_target):
        print('[DDPG] Loading Actor Weights')
        self.critic.model.load_weights(path)
        self.critic.target_model.load_weights(path_target)

    def plot_y(self, y, name):
        time = np.arange(y.shape[0])
        fig, axes = plt.subplots(4,1, figsize = (5,20))
        for i in range(4):
            axes[i].plot(time, y[:, 3*i], color = 'r', label = 'ankle')
            axes[i].plot(time, y[:, 3*i + 1], color = 'g', label = 'knee')
            axes[i].plot(time, y[:, 3*i + 2], color = 'b', label = 'hip')
            axes[i].legend()
        fig.savefig(name)
        plt.close()

    def _add_noise(self, action):
        # noise theta and sigma scaled by 0.1 for exp5
        self._noise[0] = max(self.epsilon, 0) * self.OU.function(
            action[0],
            0.0,
            0.15,
            0.2
        )
        self._noise[1] = max(self.epsilon, 0) * self.OU.function(
            action[1],
            0.0,
            0.15,
            0.2
        )
        self._action[0] = action[0] + self._noise[0]
        self._action[1] = action[1] + self._noise[1]

    def get_batch(self):
        batch = self.replay_buffer.get_next(
            self.params['BATCH_SIZE']
        )
        batch_size = len(batch)
        states = [[] for i in range(
            len(
                self.params['observation_spec']
            )
        )]
        actions = [[] for i in range(
            len(
                self.params['action_spec']
            )
        )]
        actor_recurrent_states = [[] for i in range(len(
            self._actor_recurrent_state
        )+ 1)]
        params = [[] for i in range(len(self._params))]
        next_states = [[] for i in range(
            len(
                self.params['observation_spec']
            )
        )]
        next_actor_recurrent_states = [
            [] for i in range(len(
                self._actor_recurrent_state
            ) + 1)
        ]
        rewards = []
        step_types = []
        def print_shape(item, level):
            if not isinstance(item, list):
                print(item.shape)
                print(level)
            else:
                for l in item:
                    print_shape(l, level + 1)
        for item in batch:
            state = [[] for i in range(len(states))]
            action = [[] for i in range(len(actions))]
            reward = []
            param = [[] for i in range(len(params))]
            next_state = [[] for i in range(len(next_states))]
            step_type = []
            for j, st in enumerate(item[0][:-1]):
                for k, s in enumerate(st):
                    state[k].append(s)
                    next_state[k].append(s)
                    if j == 0 and k == len(states) - 1:
                        actor_recurrent_states[0].append(s)
                        next_actor_recurrent_states[0].append(s)
            for k, s in enumerate(item[0][-1]):
                next_state[k].append(s)
            for j, r in enumerate(item[1][0]):
                actor_recurrent_states[j + 1].append(r)
                next_actor_recurrent_states[j + 1].append(r)
            for j, ac in enumerate(item[2]):
                for k, a in enumerate(ac):
                    action[k].append(a)
            for j, prm in enumerate(item[3]):
                for k, p in enumerate(prm):
                    param[k].append(p)
            for j, rw in enumerate(item[4]):
                reward.append(tf.expand_dims(tf.expand_dims(
                    rw, 0
                ), 0))
            state = [tf.expand_dims(tf.concat(s, 0), 0) for s in state]
            action = [tf.expand_dims(tf.concat(a, 0), 0) for a in action]
            param = [tf.expand_dims(tf.concat(p, 0), 0) for p in param]
            reward = tf.expand_dims(tf.concat(reward, 0), 0)
            next_state = [tf.expand_dims(tf.concat(s, 0), 0) \
                for s in next_state]
            for i in range(len(states)):
                states[i].append(state[i])
            for i in range(len(actions)):
                actions[i].append(action[i])
            for i in range(len(params)):
                params[i].append(param[i])
            for i in range(len(next_states)):
                next_states[i].append(next_state[i])
            rewards.append(reward)

        states = [tf.concat(state, 0) for state in states]
        actor_recurrent_states = [tf.concat(ars, 0) \
            for ars in actor_recurrent_states]
        actions = [tf.concat(action, 0) for action in actions]
        params = [tf.concat(param, 0) for param in params]
        rewards = tf.concat(rewards, 0)
        next_states = [tf.concat(state, 0) for state in next_states]
        next_actor_recurrent_states = [tf.concat(ars, 0) \
            for ars in next_actor_recurrent_states]
        return states, actor_recurrent_states, actions, \
            params, rewards, next_states, \
            next_actor_recurrent_states, step_types, batch_size

    def load_pretrain_dataset(self):
        Y = np.load(
            'data/pretrain/Y.npy',
            allow_pickle = True,
            fix_imports=True
        )
        num_steps = Y.shape[1] // self.params['rnn_steps']
        Y = Y[:, num_steps * self.params['rnn_steps'], :]
        Y = np.reshape(Y, (
            Y.shape[0], num_steps, self.params['rnn_steps'], Y.shape[-1]
        ))
        time.sleep(3)
        num_data = Y.shape[0]
        Y = tf.convert_to_tensor(Y)
        F = tf.convert_to_tensor(np.load('data/pretrain/F.npy', allow_pickle = True, fix_imports=True))
        time.sleep(3)
        MU = tf.convert_to_tensor(np.load('data/pretrain/MU.npy', allow_pickle = True, fix_imports=True))
        MEAN = tf.convert_to_tensor(np.load('data/pretrain/MEAN.npy', allow_pickle = True, fix_imports=True))
        X = []
        for j in range(len(self.params['observation_spec'])):
            X.append(
                tf.data.Dataset.from_tensor_slices(
                    tf.convert_to_tensor(
                        np.load('data/pretrain/X_{j}.npy'.format(j=j), allow_pickle = True, fix_imports=True)
                    )
                )
            )
        if num_data == self.params['num_data']:
            self.num_data = self.params['num_data']
        else:
            self.num_data = num_data
        self.num_batches = num_data//self.params['pretrain_bs']
        Y = tf.data.Dataset.from_tensor_slices(Y)
        F = tf.data.Dataset.from_tensor_slices(F)
        MU = tf.data.Dataset.from_tensor_slices(MU)
        MEAN = tf.data.Dataset.from_tensor_slices(MEAN)
        Y = tf.data.Dataset.zip((Y, F, MU, MEAN))
        X = tf.data.Dataset.zip(tuple(X))
        dataset = tf.data.Dataset.zip((X, Y))
        dataset = dataset.shuffle(self.num_data).batch(
            self.params['pretrain_bs'],
            drop_remainder=True
        ).prefetch(tf.data.AUTOTUNE)
        return dataset

    def _pretrain_actor(self, x, y, W = [1,1,1,1]):
        with tf.GradientTape(persistent=False) as tape:
            out, osc, omega, a, b, state, z_out, combine_state, \
                omega_state = self.actor.model(x)
            loss_action = self.action_mse(y[0], out)
            loss_omega = self.omega_mse(y[1], omega)
            loss_a = self.a_mse(y[2], a)
            loss_b = self.b_mse(y[3], b)
            loss = W[0] * loss_action + W[1] * loss_a + \
                W[2] * loss_b + W[3] * loss_omega

        grads = tape.gradient(
            loss,
            self.actor.model.trainable_variables
        )
        self.pretrain_actor_optimizer.apply_gradients(
            zip(
                grads,
                self.actor.model.trainable_variables
            )
        )
        return loss, [loss_action, loss_omega, loss_a, loss_b]

    def test_actor(self, path):
        print('[Actor] Starting Actor Test')
        step, (x, y) = next(enumerate(self.load_dataset()))
        y_pred = self.actor.model(x)
        bs = y_pred[0].shape[0]
        action_dim = y_pred[0].shape[-1]
        steps = y_pred[0].shape[2]
        max_steps = y_pred[0].shape[1]
        shape = (bs, steps * max_steps, action_dim)
        y_pred = y_pred[0] * tf.repeat(
            tf.expand_dims(
                y_pred[3], 2),
                steps,
                2
        ) + tf.repeat(
            tf.expand_dims(
                y_pred[4],
                2
            ),
            steps,
            2
        )
        y_pred = tf.reshape(y_pred, shape)
        bs = y[0].shape[0]
        action_dim = y[0].shape[-1]
        steps = y[0].shape[2]
        max_steps = y[0].shape[1]
        shape = (bs, steps * max_steps, action_dim)
        y = y[0] * tf.repeat(
            tf.expand_dims(
                y[2],
                2
            ),
            steps,
            2
        ) + tf.repeat(
            tf.expand_dims(
                y[3],
                2
            ),
            steps,
            2
        )
        y = tf.reshape(y, shape)
        fig, ax = plt.subplots(4,1, figsize = (5,20))
        for i in range(4):
            ax[i].plot(y_pred[0][:,3*i], 'b', label = 'ankle')
            ax[i].plot(y_pred[0][:,3*i + 1], 'g', label = 'knee')
            ax[i].plot(y_pred[0][:,3*i + 2], 'r', label = 'hip')
            ax[i].legend()
        fig.savefig(os.path.join(path,'y_pred.png'))
        fig, ax = plt.subplots(4,1, figsize = (5,20))
        for i in range(4):
            ax[i].plot(y[0][:,3*i], 'b', label = 'ankle real')
            ax[i].plot(y[0][:,3*i + 1], 'g', label = 'knee real')
            ax[i].plot(y[0][:,3*i + 2], 'r', label = 'hip real')
            ax[i].legend()
        fig.savefig(os.path.join(path, 'y.png'))
        print('[Actor] Finishing Actor Test')

    def _pretrain_loop(
        self,
        grad_update,
        experiment,
        checkpoint_dir,
        name,
        epochs = None,
        start = 0,
        W = [1.0, 1.0, 1.0, 1.0]
    ):
        self.test_actor(os.path.join(
            checkpoint_dir,
            'exp{exp}'.format(exp = experiment),
            name
        ))
        if epochs is None:
            epochs = self.params['train_episode_count']
        total_loss = 0.0
        avg_loss = 0.0
        prev_loss = 1e20
        history_loss = []
        history_loss_action = []
        history_loss_a = []
        history_loss_b = []
        history_loss_omega = []
        path = os.path.join(checkpoint_dir, 'exp{ex}'.format(ex = experiment))
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, name)
        if not os.path.exists(path):
            os.mkdir(path)
        if start != 0:
            pkl = open(os.path.join(path, 'loss_{ex}_{name}_{ep}.pickle'.format(
                name = name,
                ex = experiment,
                ep = start - 1
            )), 'rb')
            history_loss = pickle.load(pkl)
            pkl.close()

            pkl = open(os.path.join(path, 'loss_action_{ex}_{name}_{ep}.pickle'.format(
                name = name,
                ex = experiment,
                ep = start - 1
            )), 'rb')
            history_loss_action = pickle.load(pkl)
            pkl.close()

            pkl = open(os.path.join(path, 'loss_a_{ex}_{name}_{ep}.pickle'.format(
                name = name,
                ex = experiment,
                ep = start - 1
            )), 'rb')
            history_loss_a = pickle.load(pkl)
            pkl.close()

            pkl = open(os.path.join(path, 'loss_b_{ex}_{name}_{ep}.pickle'.format(
                name = name,
                ex = experiment,
                ep = start - 1
            )), 'rb')
            history_loss_b = pickle.load(pkl)
            pkl.close()

            pkl = open(os.path.join(path, 'loss_omega_{ex}_{name}_{ep}.pickle'.format(
                name = name,
                ex = experiment,
                ep = start - 1
            )), 'rb')
            history_loss_omega = pickle.load(pkl)
            pkl.close()
        dataset = self.load_dataset()
        print('[Actor] Dataset {ds}'.format(ds = dataset))
        print('[Actor] Starting Actor Pretraining')
        for episode in range(start, epochs):
            print('[Actor] Starting Episode {ep}'.format(ep = episode))
            total_loss = 0.0
            total_loss_action = 0.0
            total_loss_a = 0.0
            total_loss_b = 0.0
            total_loss_omega = 0.0
            start = time.time()
            num = 0
            for step, (x, y) in enumerate(dataset):
                loss, [loss_action, loss_omega, loss_a, loss_b] = \
                    grad_update(x, y, W)
                loss = loss.numpy()
                loss_action = loss_action.numpy()
                loss_omega = loss_omega.numpy()
                loss_a = loss_a.numpy()
                loss_b = loss_b.numpy()
                print('[Actor] Episode {ep} Step {st} Loss: {loss}'.format(
                    ep = episode,
                    st = step,
                    loss = loss
                ))
                total_loss += loss
                total_loss_action += loss_action
                total_loss_a += loss_a
                total_loss_b += loss_b
                total_loss_omega += loss_omega
                num += 1
                if step > 100:
                    break
            end = time.time()
            avg_loss = total_loss / num
            avg_loss_action = total_loss_action / num
            avg_loss_a = total_loss_a / num
            avg_loss_b = total_loss_b / num
            avg_loss_omega = total_loss_omega / num
            print('-------------------------------------------------')
            print('[Actor] Episode {ep} Average Loss: {l}'.format(
                ep = episode,
                l = avg_loss
            ))
            print('[Actor] Learning Rate: {lr}'.format(
                lr = self.pretrain_actor_optimizer.lr((episode + 1) * 5))
            )
            print('[Actor] Epoch Time: {time}s'.format(time = end - start))
            print('-------------------------------------------------')
            history_loss.append(avg_loss)
            history_loss_action.append(avg_loss_action)
            history_loss_a.append(avg_loss_a)
            history_loss_b.append(avg_loss_b)
            history_loss_omega.append(avg_loss_omega)
            if episode % 3 == 0:
                if math.isnan(avg_loss):
                    break
                pkl = open(os.path.join(path, 'loss_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(history_loss, pkl)
                pkl.close()
                fig1, ax1 = plt.subplots(1, 1, figsize = (5, 5))
                ax1.plot(history_loss)
                ax1.set_xlabel('loss')
                ax1.set_ylabel('steps')
                ax1.set_title('Total Loss')
                fig1.savefig(
                    os.path.join(
                        path,
                        'loss_{ex}_{name}_{ep}.png'.format(
                            name = name,
                            ex = experiment,
                            ep = episode
                        )
                    )
                )

                pkl = open(os.path.join(path, 'loss_action_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(history_loss_action, pkl)
                pkl.close()
                fig2, ax2 = plt.subplots(1, 1, figsize = (5, 5))
                ax2.plot(history_loss_action)
                ax2.set_xlabel('loss')
                ax2.set_ylabel('steps')
                ax2.set_title('Total Action Loss')
                fig2.savefig(
                    os.path.join(
                        path,
                        'loss_action_{ex}_{name}_{ep}.png'.format(
                            name = name,
                            ex = experiment,
                            ep = episode
                        )
                    )
                )

                pkl = open(os.path.join(path, 'loss_a_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(history_loss_a, pkl)
                pkl.close()
                fig3, ax3 = plt.subplots(1, 1, figsize = (5, 5))
                ax3.plot(history_loss_a)
                ax3.set_xlabel('loss')
                ax3.set_ylabel('steps')
                ax3.set_title('Total Amplitude Loss')
                fig3.savefig(
                    os.path.join(
                        path,
                        'loss_a_{ex}_{name}_{ep}.png'.format(
                            name = name,
                            ex = experiment,
                            ep = episode
                        )
                    )
                )

                pkl = open(os.path.join(path, 'loss_b_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(history_loss_b, pkl)
                pkl.close()
                fig4, ax4 = plt.subplots(1, 1, figsize = (5, 5))
                ax4.plot(history_loss_b)
                ax4.set_xlabel('loss')
                ax4.set_ylabel('steps')
                ax4.set_title('Total Mean Loss')
                fig4.savefig(
                    os.path.join(
                        path,
                        'loss_b_{ex}_{name}_{ep}.png'.format(
                            name = name,
                            ex = experiment,
                            ep = episode
                        )
                    )
                )

                pkl = open(os.path.join(path, 'loss_omega_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(history_loss_omega, pkl)
                pkl.close()
                fig5, ax5 = plt.subplots(1, 1, figsize = (5, 5))
                ax5.plot(history_loss_omega)
                ax5.set_xlabel('loss')
                ax5.set_ylabel('steps')
                ax5.set_title('Total Mean Loss')
                fig5.savefig(
                    os.path.join(
                        path,
                        'loss_omega_{ex}_{name}_{ep}.png'.format(
                            name = name,
                            ex = experiment,
                            ep = episode
                        )
                    )
                )

                if prev_loss < avg_loss:
                    break
                else:
                    self.actor.model.save_weights(
                        os.path.join(
                            path,
                            'actor_pretrained_{name}_{ex}_{ep}.ckpt'.format(
                                ep = episode,
                                ex = experiment,
                                name = name,
                            )
                        )
                    )
                prev_loss = avg_loss
        self.test_actor(os.path.join(
            checkpoint_dir,
            'exp{exp}'.format(exp = experiment),
            name
        ))

    def pretrain_actor(self, experiment, checkpoint_dir = 'weights/actor_pretrain'):
        path = os.path.join(checkpoint_dir, 'exp{exp}'.format(exp = experiment))
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(os.path.join(
                path, 'pretrain_actor'
            ))
        self._pretrain_loop(
            self._pretrain_actor, experiment, checkpoint_dir, 'pretrain_actor',
            W = [1.0, 1.0, 1.0, 0.01]
        )

    def learn(self, model_dir, experiment, start_epoch = 0, per = False, \
            her = False):
        if per:
            print('[DDPG] Initializing PER')
            self.replay_buffer = PER(self.params)
            raise NotImplementedError
        if her:
            print('[DDPG] Initializing HER')
            raise NotImplementedError
        print('[DDPG] Training Start')
        critic_loss = []
        total_critic_loss = []
        hist_rewards = []
        total_reward = []
        _steps_ = []
        COT = []
        stability = []
        d1 = []
        d2 = []
        d3 = []
        motion = []

        self.env.set_motion_state(self.desired_motion[0][0])
        self.current_time_step = self.env.reset()
        self._state = self.current_time_step.observation
        print('[DDPG] Starting Pretraining Test')
        self.total_reward = 0.0
        step = 0
        tot_loss = 0.0
        break_loop = False
        self.epsilon -= 1/self.params['EXPLORE']
        start = None
        self._actor_recurrent_state = self.actor.recurrent_state_init
        while(step < 2 and not break_loop):
            start = time.perf_counter()
            out, osc, omega, a, b, state, z_out, combine_state, omega_state = \
                self.actor.model.layers[-1].rnn_cell(
                    self._state + self._actor_recurrent_state
                )
            self._params = [a, b]
            action_original = [out, osc]
            self._add_noise(action_original)
            if math.isnan(np.sum(self._action[0].numpy())):
                print('[DDPG] Action value NaN. Ending Episode')
                break_loop = True
            steps = self._action[0].shape[1]
            action = self._action[0] * tf.repeat(
                tf.expand_dims(self._params[0], 1),
                steps,
                axis = 1
            ) + tf.repeat(
                tf.expand_dims(self._params[1], 1),
                steps,
                axis = 1
            )
            try:
                self.current_time_step = self.env.step(
                    [action, self._action[1]],
                    self.desired_motion[0][step + 1]
                )
            except FloatingPointError:
                print('[DDPG] Floating Point Error in reward computation')
                break_loop = True
                continue
            self._state = self.current_time_step.observation
            self.actor_recurrent_state = [combine_state, omega_state]
            print('[DDPG] Step {step} Reward {reward:.5f} Time {time:.5f}'.format(
                step = step,
                reward = self.current_time_step.reward.numpy(),
                time = time.perf_counter() - start
            ))
            step += 1
        enc_goals = []
        ep = start_epoch
        if ep != 0:
            if her:
                path = os.path.join(
                    model_dir,
                    'enc_goals.pickle'
                )
                pkl = open(path, 'rb')
                enc_goals = pickle.load(pkl)
                pkl.close()
            if per:
                path = os.path.join(
                    model_dir,
                    'per_tree.pickle'
                )
                pkl = open(path, 'rb')
                tree = pickle.load(pkl)
                pkl.close()
                self.replay_buffer.set_priority_tree(tree)
            data_path = os.path.join(
                model_dir,
                'data.pickle'
            )
            pkl = open(data_path, 'rb')
            buff = pickle.load(pkl)
            pkl.close()
            self.replay_buffer.set_buffer(buff)
            self.actor.model.load_weights(
                os.path.join(
                    model_dir,
                    'actor',
                    'model',
                    'model_ep{ep}.ckpt'.format(
                        ep = ep,
                    )
                )
            )
            self.actor.target_model.load_weights(
                os.path.join(
                    model_dir,
                    'actor',
                    'target',
                    'target_model_ep{ep}.ckpt'.format(
                        ep = ep,
                    )
                )
            )
            self.critic.model.load_weights(
                os.path.join(
                    model_dir,
                    'critic',
                    'model',
                    'model_ep{ep}.ckpt'.format(
                        ep = ep,
                    )
                )
            )
            self.critic.target_model.load_weights(
                os.path.join(
                    model_dir,
                    'critic',
                    'target',
                    'target_model_ep{ep}.ckpt'.format(
                        ep = ep,
                    )
                )
            )
            pkl = open(os.path.join(
                model_dir,
                'rewards_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            hist_rewards = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'total_reward_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            total_reward = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'critic_loss_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            critic_loss = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'total_critic_loss_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            total_critic_loss = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'COT_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            COT = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'motion_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            motion = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'stability_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            stability = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'd1_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            d1 = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'd2_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            d2 = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'd3_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            d3 = pickle.load(pkl)
            pkl.close()
        goal_id = np.random.randint(0, len(self.desired_motion))
        desired_motion = self.desired_motion[goal_id]
        print(desired_motion)
        while ep < self.params['train_episode_count']:
            enc_goals.append(desired_motion[0])
            self._action = self.env._action_init
            self._noise = self._noise_init
            self.env.set_motion_state(desired_motion[0])
            self.current_time_step = self.env.reset()
            self._state = self.current_time_step.observation
            self._actor_recurrent_state = self.actor.recurrent_state_init
            print('[DDPG] Starting Episode {i}'.format(i = ep))
            self._state = self.current_time_step.observation
            self.total_reward = 0.0
            step = 0
            tot_loss = 0.0
            break_loop = False
            self.epsilon -= 1/self.params['EXPLORE']
            start = None
            observations = []
            observations.append(self._state)
            actor_recurrent_states = []
            actor_recurrent_states.append(self._actor_recurrent_state)
            actions = []
            rewards = []
            params = []
            while(step < self.params['max_steps'] and not break_loop):
                penalty = tf.convert_to_tensor(0.0, dtype = tf.dtypes.float32)
                start = time.perf_counter()
                out, osc, omega, a, b, state, z_out, combine_state, omega_state = \
                    self.actor.model.layers[-1].rnn_cell(
                        self._state + self._actor_recurrent_state
                    )
                self._params = [a, b]
                action_original = [out, osc]
                self._add_noise(action_original)
                if math.isnan(np.sum(self._action[0].numpy())):
                    print('[DDPG] Action value NaN. Ending Episode')
                    penalty += tf.convert_to_tensor(-5.0, dtype = tf.dtypes.float32)
                    self._action[0] = tf.zeros_like(self._action[0])
                steps = self._action[0].shape[1]
                action = self._action[0] * tf.repeat(
                    tf.expand_dims(self._params[0], 1),
                    steps,
                    axis = 1
                ) + tf.repeat(
                    tf.expand_dims(self._params[1], 1),
                    steps,
                    axis = 1
                )
                try:
                    last_step = False
                    first_step = False
                    if step == 0:
                        first_step = True
                    if step < self.params['max_steps'] - 1:
                        last_step = False
                    else:
                        last_step = True
                    self.current_time_step = self.env.step(
                        [action, self._action[1]],
                        desired_motion[step + 1],
                        last_step = last_step,
                        first_step = first_step
                    )
                except FloatingPointError:
                    print('[DDPG] Floating Point Error in reward computation')
                    penalty += tf.convert_to_tensor(-5.0, dtype = tf.dtypes.float32)
                r = self.current_time_step.reward + penalty
                motion.append(self.env.quadruped.r_motion)
                COT.append(self.env.quadruped.COT)
                rewards.append(r)
                actions.append(self._action)
                params.append(self._params)
                self._actor_recurrent_state = [combine_state, omega_state]
                actor_recurrent_states.append(self._actor_recurrent_state)
                self._state = self.current_time_step.observation
                observations.append(self._state)
                stability.append(self.env.quadruped.stability)
                d1.append(self.env.quadruped.d1)
                d2.append(self.env.quadruped.d2)
                d3.append(self.env.quadruped.d3)
                hist_rewards.append(r.numpy())
                self.total_reward += r.numpy()
                print('[DDPG] Episode {ep} Step {step} Reward {reward:.5f} Time {time:.5f}'.format(
                    ep = ep,
                    step = step,
                    reward = r.numpy(),
                    time = time.perf_counter() - start
                ))
                step += 1
                if self.current_time_step.step_type == \
                    tfa.trajectories.time_step.StepType.LAST:
                    break_loop = True
            experience = [
                observations,
                actor_recurrent_states,
                actions,
                params,
                rewards
            ]
            self.replay_buffer.add_batch(experience)
            start = time.perf_counter()
            states, actor_recurrent_states, actions, \
                params, rewards, next_states, \
                next_actor_recurrent_states, step_types, batch_size = \
                self.get_batch()
            out, osc, omega, a, b, state, z_out, combine_state, omega_state = \
                self.actor.target_model(
                    next_states[:-1] + next_actor_recurrent_states
                )
            ac = [out, osc]
            recurrent_state = [tf.repeat(
                rci,
                batch_size,
                0
            ) for rci in self.critic.recurrent_state_init]
            inputs = next_states + ac + [a, b] + recurrent_state
            target_q_values, _ = self.critic.target_model(inputs)
            y = tf.concat([
                rewards[:, :-1] + \
                    self.params['GAMMA'] * target_q_values[:, 1:-1],
                rewards[:, -1:]
            ], 1)
            loss = self.critic.train(states, actions, params[0], \
                params[1], recurrent_state,  y)
            critic_loss.append(loss.numpy())
            tot_loss += loss.numpy()

            out, osc, omega, a, b, state, z_out, combine_state, omega_state = \
                    self.actor.model(states[:-1] + actor_recurrent_states)
            a_for_grad = [out, osc]
            q_grad = self.critic.q_grads(states, a_for_grad, \
                a, b, recurrent_state)
            self.actor.train(states, actor_recurrent_states, q_grad)
            print('[DDPG] Critic Loss {loss}'.format(
                loss = loss.numpy(),
            ))
            print('[DDPG] Total Reward {reward:.5f} Avg Critic Loss {loss:.5f} Time {time:.5f}'.format(
                reward = self.total_reward,
                loss = tot_loss,
                time = time.perf_counter() - start
            ))
            if ep % self.params['TEST_AFTER_N_EPISODES'] == 0:
                self.save(model_dir, ep, hist_rewards, total_reward, \
                    total_critic_loss, critic_loss, COT, motion, \
                    stability, d1, d2, d3)
            _steps_.append(step + 1)
            total_reward.append(self.total_reward)
            total_critic_loss.append(tot_loss)
            ep += 1
            print('[DDPG] Starting Next Episode')

    def save(self, model_dir, ep, rewards, total_reward, total_critic_loss, \
            critic_loss, COT, motion, stability, d1, d2, d3, tree = None, enc_goals = None):
        print('[DDPG] Saving Data')
        data_path = os.path.join(
            model_dir,
            'data.pickle'
        )
        if os.path.exists(data_path):
            os.remove(data_path)
        pkl = open(data_path, 'wb')
        pickle.dump(self.replay_buffer.buffer, pkl)
        pkl.close()
        if tree is not None:
            print('[DDPG] Saving PER priorities')
            path = os.path.join(
                model_dir,
                'per_tree.pickle'
            )
            if os.path.exists(path):
                os.remove(path)
            pkl = open(path, 'wb')
            pickle.dump(tree, pkl)
            pkl.close()
        if enc_goals is not None:
            print('[DDPG] Saving HER goals')
            path = os.path.join(
                model_dir,
                'enc_goals.pickle'
            )
            pkl = open(path, 'wb')
            pickle.dump(enc_goals, pkl)
            pkl.close()
        print('[DDPG] Saving Model')
        self.actor.model.save_weights(
            os.path.join(
                model_dir,
                'actor',
                'model',
                'model_ep{ep}.ckpt'.format(
                    ep = ep,
                )
            )
        )

        self.actor.target_model.save_weights(
            os.path.join(
                model_dir,
                'actor',
                'target',
                'target_model_ep{ep}.ckpt'.format(
                    ep = ep,
                )
            )
        )

        self.critic.model.save_weights(
            os.path.join(
                model_dir,
                'critic',
                'model',
                'model_ep{ep}.ckpt'.format(
                    ep = ep,
                )
            )
        )

        self.critic.model.save_weights(
            os.path.join(
                model_dir,
                'critic',
                'target',
                'target_model_ep{ep}.ckpt'.format(
                    ep = ep,
                )
            )
        )

        pkl = open(os.path.join(
            model_dir,
            'rewards_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(rewards, pkl)
        pkl.close()
        fig1, ax1 = plt.subplots(1,1,figsize = (5,5))
        ax1.plot(rewards)
        ax1.set_ylabel('reward')
        ax1.set_xlabel('steps')
        fig1.savefig(os.path.join(
            model_dir,
            'rewards_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
           'total_reward_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(total_reward, pkl)
        pkl.close()
        fig2, ax2 = plt.subplots(1,1,figsize = (5,5))
        ax2.plot(total_reward)
        ax2.set_ylabel('total reward')
        ax2.set_xlabel('episodes')
        fig2.savefig(os.path.join(
            model_dir,
            'total_reward_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'critic_loss_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(critic_loss, pkl)
        pkl.close()
        fig3, ax3 = plt.subplots(1,1,figsize = (5,5))
        ax3.plot(critic_loss)
        ax3.set_ylabel('critic loss')
        ax3.set_xlabel('steps')
        fig3.savefig(os.path.join(
            model_dir,
            'critic_loss_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'total_critic_loss_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(total_critic_loss, pkl)
        pkl.close()
        fig4, ax4 = plt.subplots(1,1,figsize = (5,5))
        ax4.plot(total_critic_loss)
        ax4.set_ylabel('total critic loss')
        ax4.set_xlabel('episodes')
        fig4.savefig(os.path.join(
            model_dir,
            'total_critic_loss_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'COT_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(COT, pkl)
        pkl.close()
        fig9, ax9 = plt.subplots(1,1,figsize = (5,5))
        ax9.plot(COT)
        ax9.set_ylabel('COT')
        ax9.set_xlabel('steps')
        fig9.savefig(os.path.join(
            model_dir,
            'COT_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'motion_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(motion, pkl)
        pkl.close()
        fig10, ax10 = plt.subplots(1,1,figsize = (5,5))
        ax10.plot(motion)
        ax10.set_ylabel('motion')
        ax10.set_xlabel('steps')
        fig10.savefig(os.path.join(
            model_dir,
            'motion_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'stability_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(stability, pkl)
        pkl.close()
        fig11, ax11 = plt.subplots(1,1,figsize = (5,5))
        ax11.plot(stability)
        ax11.set_ylabel('stability')
        ax11.set_xlabel('steps')
        fig11.savefig(os.path.join(
            model_dir,
            'stability_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'd1_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(d1, pkl)
        pkl.close()
        fig12, ax12 = plt.subplots(1,1,figsize = (5,5))
        ax12.plot(d1)
        ax12.set_ylabel('d1')
        ax12.set_xlabel('steps')
        fig12.savefig(os.path.join(
            model_dir,
                'd1_ep{ep}.png'.format(
                    ep = ep
                )
            )
        )

        pkl = open(os.path.join(
            model_dir,
            'd2_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(d2, pkl)
        pkl.close()
        fig13, ax13 = plt.subplots(1,1,figsize = (5,5))
        ax13.plot(d2)
        ax13.set_ylabel('d2')
        ax13.set_xlabel('steps')
        fig13.savefig(os.path.join(
            model_dir,
            'd2_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'd3_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(d3, pkl)
        pkl.close()
        fig14, ax14 = plt.subplots(1,1,figsize = (5,5))
        ax14.plot(d3)
        ax14.set_ylabel('d3')
        ax14.set_xlabel('steps')
        fig14.savefig(os.path.join(
            model_dir,
            'd3_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        plt.close('all')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type = int,
        help = 'ID of experiment being performaed'
    )
    parser.add_argument(
        '--out_path',
        type = str,
        help = 'Path to output directory'
    )
    parser.add_argument(
        '--start',
        type = int,
        help = 'start epoch',
        default = 0
    )

    parser.add_argument(
        "--per",
        type = str2bool,
        nargs = '?',
        const = True,
        default = False,
        help = "Toggle PER"
    )

    parser.add_argument(
        "--her",
        type = str2bool,
        nargs = '?',
        const = True,
        default = False,
        help = "Toggle HER"
    )
    args = parser.parse_args()
    learner = Learner(params, args.experiment, True)
    #learner.pretrain_actor(args.experiment, args.out_path)
    """
    path = os.path.join(args.out_path, 'exp{exp}'.format(
        exp=args.experiment
    ))

    if not os.path.exists(path):
        os.mkdir(path)

    actor_path = os.path.join(path, 'actor')
    if not os.path.exists(actor_path):
        os.mkdir(actor_path)
        os.mkdir(os.path.join(actor_path, 'model'))
        os.mkdir(os.path.join(actor_path, 'target'))

    critic_path = os.path.join(path, 'critic')
    if not os.path.exists(critic_path):
        os.mkdir(critic_path)
        os.mkdir(os.path.join(critic_path, 'model'))
        os.mkdir(os.path.join(critic_path, 'target'))

    learner.learn(
        path,
        experiment = args.experiment,
        start_epoch = args.start,
        per = args.per,
        her = args.her
    )
    """
