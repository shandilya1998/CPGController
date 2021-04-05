from rl.constants import params
import rospy
from rl.net import ActorNetwork, CriticNetwork
from rl.env import Env
from rl.replay_buffer import ReplayBuffer, OU
import tf_agents as tfa
import tensorflow as tf
import numpy as np
from gait_generation.gait_generator import Signal
from tf_agents.trajectories.time_step import TimeStep, time_step_spec
from tqdm import tqdm
import pickle
import os
from frequency_analysis import frequency_estimator
import time
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib
import math

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class SignalDataGen:
    def __init__(self, params):
        self.Tst = params['Tst']
        self.Tsw = params['Tsw']
        self.theta_h = params['theta_h']
        self.theta_k = params['theta_k']
        self.N = params['rnn_steps']
        self.params = params
        self.signal_gen = Signal(
            self.N + 1,
            self.params['dt']
        )
        self.dt = self.params['dt']
        self.data = []
        self.num_data = 0

    def set_N(self, N, create_data = False):
        self.N = N
        self.signal_gen = Signal(
            self.N + 1,
            self.params['dt']
        )
        if create_data:
            self._create_data()
        else:
            self.num_data = self.params['num_data'] // self.params['rnn_steps']

    def get_ff(self, signal, ff_type = 'fft'):
        if ff_type == 'fft':
            return frequency_estimator.freq_from_fft(signal, 1/self.dt)
        elif ff_type == 'autocorr':
            return frequency_estimator.freq_from_autocorr(signal, 1/self.dt)

    def _create_data(self):
        """
            Turning Behaviour is to be learnt by RL
        """
        print('[Actor] Creating Data.')
        self.data = []
        deltas = [0]#, 3]#, -3]
        delta = []
        for i in range(len(deltas)):
            for j in range(len(deltas)):
                delta.append([
                    deltas[i],
                    deltas[j],
                ])

        """
            Data for straight ahead
        """
        for d in tqdm(delta):
            for tst, tsw, theta_h, theta_k in zip(
                self.Tst,
                self.Tsw,
                self.theta_h,
                self.theta_k
            ):
                tsw = tsw + d[0]
                tst = tst + d[1]
                self.signal_gen.build(tsw, tst, theta_h, theta_k)
                signal, _ = self.signal_gen.get_signal()
                signal = signal[:, 1:].astype(np.float32)
                v = self.signal_gen.compute_v((0.1+0.015)*2.2)
                motion = np.array([0, -1, 0, 0, -1 * v ,0], dtype = np.float32)
                freq = np.float32(self.get_ff(signal[:, 2], 'fft'))
                self.data.append(
                    [signal, motion, freq]
                )
        self.num_data = len(self.data)
        print('[Actor] Number of Data Points: {num}'.format(
            num = self.num_data)
        )

    def preprocess(self, signal):
        mean = np.mean(signal, axis = 0)
        mu = np.abs(signal).max(axis = 0)
        signal = (signal - mean) / mu
        return signal, mean, mu

    def generator(self):
        for batch in range(self.num_data):
            y, x, f = self.data[batch]
            yield y, x, f

class Learner():
    def __init__(self, params, experiment, create_data = False):
        np.seterr(all='ignore')
        self.params = params
        self.actor = ActorNetwork(params)
        self.critic = CriticNetwork(params)
        self.replay_buffer = ReplayBuffer(params)
        self.time_step_spec = tfa.trajectories.time_step.time_step_spec(
            observation_spec = self.params['observation_spec'],
            reward_spec = self.params['reward_spec']
        )
        self.env = Env(
            self.time_step_spec,
            self.params,
            experiment
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
        self.signal_gen = SignalDataGen(params)
        self.signal_gen.set_N(3 * self.params['rnn_steps'], create_data)
        self.pretrain_osc_mu = np.ones((
            1,
            self.params['units_osc']
        ), dtype = np.float32)
        self.mse_mu = tf.keras.losses.MeanSquaredError()
        self.mse_mean = tf.keras.losses.MeanSquaredError()
        self.mse_omega = tf.keras.losses.MeanSquaredError()
        self.dt = self.params['dt']
        if create_data:
            self.create_dataset()
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001,
            decay_steps=60,
            decay_rate=0.95
        )
        self.pretrain_actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate = lr_schedule
        )
        self.pretrain_dataset = self.load_dataset()
        i, (x, y) = next(enumerate(self.pretrain_dataset))
        self.osc = x[-1][0]
        self.desired_motion = np.repeat(
            np.expand_dims(x[0][0], 0),
            self.params['max_steps'] + 1,
            0
        )
        self._state = [
            self.env.quadruped.motion_state,
            self.env.quadruped.robot_state,
            self.env.quadruped.osc_state
        ]
        self._action = None
        matplotlib.use('Agg')
        physical_devices = tf.config.list_physical_devices('GPU')
        np.seterr(all='raise')
        print('[Actor] GPU>>>>>>>>>>>>')
        print('[Actor] {lst}'.format(lst = physical_devices))

    def set_desired_motion(self, motion):
        self.desired_motion = motion

    def print_grads(self, name, grads):
        print('[Actor] {name} >>>>>>>>>>>>'.format(name=name))
        print(grads)

    def _hopf_oscillator(self, omega, mu, b, z):
        rng = np.arange(1, self.params['units_osc'] + 1)
        x, y = z[:self.params['units_osc']], z[self.params['units_osc']:]
        rng = omega * rng
        mod = (mu - (np.square(x) + np.square(y)))
        x = x + (
            np.multiply(
                mod,
                x
            ) - np.multiply(
                rng, 
                y
            )
        ) * self.dt + b
        y = y + (
            np.multiply(
                mod,
                y
            ) + np.multiply(
                rng,
                x
            )
        ) * self.dt + b
        return np.concatenate([x, y], -1)

    def create_dataset(self):
        self.num_batches = self.params['num_data']//self.params['pretrain_bs']
        self.env.quadruped.reset()
        F = []
        MU = []
        Y = []
        MEAN = []
        X = [[] for j in range(len(self.params['observation_spec']))]
        for y, x, f in tqdm(self.signal_gen.generator()):
            f = f * 2 * np.pi
            y = y * np.pi / 180
            for i in range(self.params['rnn_steps']):
                ac = y[i]
                if np.isinf(ac).any():
                    print('Inf in unprocessed')
                y_, mean, mu = self.signal_gen.preprocess(y)
                if np.isinf(y_).any():
                    print('Inf')
                    break
                mu = mu / (np.pi/3)
                actions = np.expand_dims(y_[i + 1: i + 1 + self.params['rnn_steps']], 0)
                self.env.quadruped.all_legs.move(ac)
                self.env.quadruped.set_motion_state(x)
                _state = self.env.quadruped.get_state_tensor()
                for j, s in enumerate(_state):
                    X[j].append(s)
                Y.append(actions)
                F.append(np.array([[f]], dtype = np.float32))
                MU.append(np.expand_dims(mu, 0))
                self.env.quadruped._hopf_oscillator(
                    f,
                    np.ones((self.params['units_osc'],)),
                    np.zeros((self.params['units_osc'],)),
                )
                MEAN.append(np.expand_dims(mean, 0))
            self.env.quadruped.reset()

        for j in range(len(X)):
            X[j] = np.concatenate(X[j], axis = 0)
        Y = np.concatenate(Y, axis = 0)
        F = np.concatenate(F, axis = 0)
        MU = np.concatenate(MU, axis = 0)
        MEAN = np.concatenate(MEAN, axis = 0)
        print('[Actor] Y Shape : {sh}'.format(sh=Y.shape))
        np.save('data/pretrain/Y.npy', Y, allow_pickle = True, fix_imports=True)
        time.sleep(3)
        np.save('data/pretrain/F.npy', F, allow_pickle = True, fix_imports=True)
        time.sleep(3)
        np.save('data/pretrain/MU.npy', MU,allow_pickle = True,fix_imports=True)
        time.sleep(3)
        np.save(
            'data/pretrain/MEAN.npy',
            MEAN,
            allow_pickle = True,
            fix_imports=True
        )
        time.sleep(3)
        for j in range(len(X)):
            time.sleep(3)
            np.save('data/pretrain/X_{j}.npy'.format(j=j), X[j], allow_pickle = True, fix_imports=True)

    def load_dataset(self):
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

    def _pretrain_loop(
        self, 
        grad_update, 
        experiment, 
        checkpoint_dir, 
        name, 
        epochs = None, 
        start = 0
    ):
        if epochs is None:
            epochs = self.params['train_episode_count']
        total_loss = 0.0
        avg_loss = 0.0
        prev_loss = 1e20
        history_loss = []
        history_loss_action = []
        history_loss_mu = []
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

            pkl = open(os.path.join(path, 'loss_mu_{ex}_{name}_{ep}.pickle'.format(
                name = name,
                ex = experiment,
                ep = start - 1
            )), 'rb')
            history_loss_mu = pickle.load(pkl)
            pkl.close()

            pkl = open(os.path.join(path, 'loss_omega_{ex}_{name}_{ep}.pickle'.format(
                name = name,
                ex = experiment,
                ep = start - 1
            )), 'rb')
            history_loss_omega = pickle.load(pkl)
            pkl.close()
        """
        dataset = tf.data.Dataset.from_generator(
            self._dataset,
            output_types = [
                [spec.dtype for spec in self.params['observation_spec']],
                self.params['action_spec'][0].dtype
            ]
        )
        """
        dataset = self.load_dataset()
        print('[Actor] Dataset {ds}'.format(ds = dataset))
        print('[Actor] Starting Actor Pretraining')
        for episode in range(start, epochs):
            print('[Actor] Starting Episode {ep}'.format(ep = episode))
            total_loss = 0.0
            total_loss_action = 0.0
            total_loss_mu = 0.0
            total_loss_omega = 0.0
            start = time.time()
            num = 0
            for step, (x, y) in enumerate(dataset):
                loss, [loss_action, loss_omega, loss_mu] = \
                    grad_update(x, y)
                loss = loss.numpy()
                print('[Actor] Episode {ep} Step {st} Loss: {loss}'.format(
                    ep = episode,
                    st = step,
                    loss = loss
                ))
                total_loss += loss
                total_loss_action += loss_action
                total_loss_mu += loss_mu
                total_loss_omega += loss_omega
                num += 1
                if step > 100:
                    break
            end = time.time()
            avg_loss = total_loss / num
            avg_loss_action = total_loss_action / num
            avg_loss_mu = total_loss_mu / num
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
            history_loss_mu.append(avg_loss_mu)
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

                pkl = open(os.path.join(path, 'loss_action_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(history_loss_action, pkl)
                pkl.close()

                pkl = open(os.path.join(path, 'loss_mu_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(history_loss_mu, pkl)
                pkl.close()

                pkl = open(os.path.join(path, 'loss_omega_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(history_loss_omega, pkl)
                pkl.close()
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

    def _pretrain_actor(self, x, y):
        with tf.GradientTape(persistent=False) as tape:
            _action, [omega, mu, mean, state] = self.actor.model(x)
            y_pred = _action[0]
            loss_mu = self.mse_mu(y[2], mu)
            loss_mean = self.mse_mean(y[3], mean)
            loss_action = self.actor._pretrain_loss(y[0], y_pred)
            loss_omega = self.mse_omega(y[1], omega)
            loss = loss_mu + loss_action + loss_mean + loss_omega

        grads = tape.gradient(
            loss,
            self.actor.model.trainable_variables
        )
        #self.print_grads(self.actor.model.trainable_variables, grads_action)
        self.pretrain_actor_optimizer.apply_gradients(
            zip(
                grads,
                self.actor.model.trainable_variables
            )
        )

        return loss, [loss_action, loss_omega, loss_mu]

    def _pretrain_encoder(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            state, omega, mu, mean = self.actor.model(x)
            loss_mu = self.mse_mu(y[2], mu)
            loss_mean = self.mse_mean(y[3], mean)
            loss_omega = self.mse_omega(y[1], omega)
            loss = loss_omega + loss_mu + loss_mean

        grads = tape.gradient(
            loss,
            self.actor.model.trainable_variables
        )
        #self.print_grads(self.actor.model.trainable_variables, grads_action)
        self.pretrain_actor_optimizer.apply_gradients(
            zip(
                grads,
                self.actor.model.trainable_variables
            )
        )
        return loss, [0.0, loss_omega, loss_mu]

    def _pretrain_actor_action_loss_only(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            _action, [omega, mu, mean, state] = self.actor.model(x)
            y_pred = _action[0]
            """
            print('state')
            print(state)
            print('omega')
            print(omega)
            print('y_pred')
            print(y_pred)
            print('osc')
            print(_action[1])
            """
            loss_mu = self.mse_mu(y[2], mu)
            loss_mean = self.mse_mean(y[3], mean)
            loss_omega = self.mse_omega(y[1], omega)
            loss_action = self.actor._pretrain_loss(y[0], y_pred)
            loss = loss_mu + loss_mean + loss_action

        grads = tape.gradient(
            loss_action,
            self.actor.model.trainable_variables
        )
        #self.print_grads(self.actor.model.trainable_variables, grads_action)
        self.pretrain_actor_optimizer.apply_gradients(
            zip(
                grads,
                self.actor.model.trainable_variables
            )
        )
        return loss, [loss_action, loss_omega, loss_mu]

    def _pretrain_segments(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            _action, [omega, mu, mean] = self.actor.model(x)
            y_pred = _action[0]
            loss_mu = self.mse_mu(y[2], mu)
            loss_mean = self.mse_mean(y[3], mean)
            loss_omega = self.mse_omega(y[1], omega)
            loss_action = self.actor._pretrain_loss(y[0], y_pred)
            loss_enc = loss_omega + loss_mu + loss_mean
            loss = loss_enc + loss_action
        vars_encoder = []
        vars_remainder = []
        for var in self.actor.model.trainable_variables:
            if 'motion' in var.name or 'omega' in var.name:
                vars_encoder.append(var)
            elif 'mean' in var.name or 'mu' in var.name:
                vars_encoder.append(var)
            elif 'robot' in var.name or 'param' in var.name:
                vars_encoder.append(var)
            else:
                vars_remainder.append(var)
        grads = tape.gradient(
            loss_enc,
            vars_encoder
        )
        #self.print_grads(self.actor.model.trainable_variables, grads_action)
        self.pretrain_actor_optimizer.apply_gradients(
            zip(
                grads,
                vars_encoder
            )
        )
        grads = tape.gradient(
            loss_action,
            vars_remainder
        )
        #self.print_grads(self.actor.model.trainable_variables, grads_action)
        self.pretrain_actor_optimizer.apply_gradients(
            zip(
                grads,
                vars_remainder
            )
        )

        return loss, [loss_action, loss_omega, loss_mu]

    def pretrain_actor(self, experiment, checkpoint_dir = 'weights/actor_pretrain'):
        encoder = self.actor.create_encoder(self.params)
        self.actor.set_model(encoder)
        self._pretrain_loop(
            self._pretrain_encoder, 
            experiment, 
            checkpoint_dir, 
            'pretrain_enc',
            epochs = 40
        )
        self.actor.model.load_weights(os.path.join(
            checkpoint_dir, 'exp23/pretrain_enc', 'actor_pretrained_pretrain_enc_23_39.ckpt'
        ))

        encoder =  self.actor.make_untrainable(self.actor.model)
        model, _, _ = self.actor.create_actor_network(params, self.actor.model)#encoder)
        self.actor.set_model(model)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001,
            decay_steps=70,
            decay_rate=0.95
        )
        self.pretrain_actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate = lr_schedule
        )

        self._pretrain_loop(
            self._pretrain_actor, experiment, checkpoint_dir, 'pretrain_actor'
        )

    def load_actor(self, path):
        print('[DDPG] Loading Actor Weights')
        self.actor.model.load_weights(path)

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
            0.015,
            0.02
        )
        self._noise[1] = max(self.epsilon, 0) * self.OU.function(
            action[1],
            0.0,
            0.015,
            0.02
        )
        self._action[0] = action[0] + self._noise[0]
        self._action[1] = action[1] + self._noise[1]

    def learn_a2c(self, model_dir, experiment):
        ep = 0

    def learn(self, model_dir, experiment):
        ep = 0
        self.epsilon = 1
        self._noise_init = [
                    tf.expand_dims(tf.zeros(
                        spec.shape,
                        spec.dtype
                    ), 0) for spec in self.env.action_spec()
                ]

        motion_state, robot_state, osc_state = \
            self.env.quadruped.get_state_tensor()


        print('[DDPG] Training Start')
        critic_loss = []
        total_critic_loss = []
        rewards = []
        total_reward = []
        _steps_ = []
        COT = []
        motion = []
        self.mass = self.env.quadruped.mass
        self.gravity = self.env.quadruped.gravity
        while ep < self.params['train_episode_count']:
            self.env.set_motion_state(self.desired_motion[0])
            self.current_time_step = self.env.reset()
            print('[DDPG] Starting Episode {i}'.format(i = ep))
            self._state = self.current_time_step.observation
            self.total_reward = 0.0
            step = 0
            tot_loss = 0.0
            break_loop = False
            while(step < self.params['max_steps'] and not break_loop):
                self.epsilon -= 1/self.params['EXPLORE']
                self._action = self.env._action_init
                self._noise = self._noise_init
                [out, osc], [omega, mu, mean, state] = self.actor.model(self._state)
                self._params = [mu, mean]
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

                self.current_time_step = self.env.step(
                    [action, self._action[1]],
                    self.desired_motion[step + 1]
                )
                motion.append(self.env.quadruped.r_motion)
                COT.append(self.env.quadruped.COT)
                start = time.time()
                experience = [
                    self._state,
                    self._action,
                    self._params,
                    self.current_time_step.reward,
                    self.current_time_step.observation,
                    self.current_time_step.step_type
                ]
                rewards.append(self.current_time_step.reward.numpy())
                self.replay_buffer.add_batch(experience)
                batch = self.replay_buffer.get_next(
                    self.params['BATCH_SIZE']
                )
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
                params = [[] for i in range(len(self._params))]
                next_states = [[] for i in range(
                    len(
                        self.params['observation_spec']
                    )
                )]
                rewards = []
                step_types = []
                for item in batch:
                    state = item[0]
                    action = item[1]
                    param = item[2]
                    rewards.append(item[3])
                    next_state = item[4]
                    step_types.append(item[5])
                    for i, s in enumerate(state):
                        states[i].append(s)
                    for i, p in enumerate(param):
                        params[i].append(p)
                    for i, a in enumerate(action):
                        actions[i].append(a)
                    for i, s in enumerate(next_state):
                        next_states[i].append(s)
                states = [tf.concat(state, 0) for state in states]
                actions = [tf.concat(action, 0) for action in actions]
                params = [tf.concat(param, 0) for param in params]
                #rewards = tf.concat(rewards, 0)
                next_states = [tf.concat(state, 0) for state in next_states]
                [out, osc],[o, m, mn, st] = self.actor.target_model(next_states)
                ac = [out, osc]
                inputs = next_states + ac + [m, mn]
                target_q_values = self.critic.target_model(inputs)
                y = [tf.repeat(reward, self.params['action_dim'])\
                    for reward in rewards]
                y = [
                    y[k] + self.params['GAMMA'] * target_q_values[k] \
                        if step_types[k] != \
                        tfa.trajectories.time_step.StepType.LAST \
                    else y[k] for k in range(len(y))
                ]
                y = tf.concat([
                    tf.expand_dims(_y, 0) for _y in y
                ], 0)
                loss = self.critic.train(states, actions, params[0], \
                    params[1], y)
                critic_loss.append(loss.numpy())
                tot_loss += loss.numpy()
                a_for_grad,[omega_,mu_,mean_, state_] = self.actor.model(states)
                q_grads = self.critic.q_grads(states, a_for_grad, \
                    mu_, mean_)
                self.actor.train(states, q_grads)
                self.actor.target_train()
                self.critic.target_train()
                self.total_reward += self.current_time_step.reward
                self._state = self.current_time_step.observation
                print('[DDPG] Episode {ep} Step {step}'.format(
                    ep = ep,
                    step = step 
                ))
                print('[DDPG] Total Reward {reward} Critic Loss {loss}'.format(
                    reward = self.current_time_step.reward,
                    loss = loss.numpy()
                ))
                step += 1
                if self.current_time_step.step_type == \
                    tfa.trajectories.time_step.StepType.LAST:
                    print('[DDPG] Starting Next Episode')
                    break_loop = True

            if ep % 3 == 0:
                self.save(model_dir, ep, rewards, total_reward, \
                    total_critic_loss, critic_loss, COT, motion)

            _steps_.append(step + 1)
            ep += 1
            total_reward.append(self.total_reward)
            total_critic_loss.append(tot_loss)

    def save(self, model_dir, ep, rewards, total_reward, total_critic_loss, \
            critic_loss, COT, motion):
        print('\n[DDPG] Saving Model')
        self.actor.model.save_weights(
            os.path.join(
                model_dir,
                'actor',
                'model_ep{ep}.ckpt'.format(
                    ep = ep,
                )
            )
        )

        self.critic.model.save_weights(
            os.path.join(
                model_dir,
                'critic',
                'model_ep{ep}.ckpt'.format(
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
        ax10.set_ylabel('motion deviation')
        ax10.set_xlabel('steps')
        fig10.savefig(os.path.join(
            model_dir,
            'motion_ep{ep}.png'.format(
                ep = ep,
            )
        ))
        plt.close('all')

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
    args = parser.parse_args()
    learner = Learner(params, args.experiment, False)
    learner.pretrain_actor(
        args.experiment,
        args.out_path,
    )
    #"""
    #learner.load_actor('weights/actor_pretrain/exp23/pretrain_actor/actor_pretrained_pretrain_actor_23_60.ckpt')
    path = os.path.join(args.out_path, 'exp{exp}'.format(
        exp=args.experiment
    ))

    if not os.path.exists(path):
        os.mkdir(path)

    actor_path = os.path.join(path, 'actor')
    if not os.path.exists(actor_path):
        os.mkdir(actor_path)

    critic_path = os.path.join(path, 'critic')
    if not os.path.exists(critic_path):
        os.mkdir(critic_path)

    learner.learn(path, experiment = args.experiment)
    #"""
