import numpy as np
import tensorflow as tf
import os
import pickle
import time
import math
from rl.rddpg_net_v2 import ActorNetwork, CriticNetwork
from rl.env import Env
from rl.constants import params
from rl.replay_buffer import ReplayBuffer, OU
import argparse
import tf_agents as tfa
import matplotlib.pyplot as plt
import matplotlib
from learn import SignalDataGen
from tqdm import tqdm
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class Learner:
    def __init__(self, params, experiment, create_data = False):
        self.params = params
        self.experiment = experiment
        self.actor = ActorNetwork(self.params)
        self.critic = CriticNetwork(self.params)
        self.time_step_spec = tfa.trajectories.time_step.time_step_spec(
            observation_spec = self.params['observation_spec'],
            reward_spec = self.params['reward_spec']
        )
        self.replay_buffer = ReplayBuffer(self.params)
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
        matplotlib.use('Agg')
        physical_devices = tf.config.list_physical_devices('GPU')
        np.seterr(all='raise')
        print('[Actor] GPU>>>>>>>>>>>>')
        print('[Actor] {lst}'.format(lst = physical_devices))
        self.p1 = 1.0
        self.epsilon = 1
        self.signal_gen = SignalDataGen(params)
        self.signal_gen.set_N(
            self.params['rnn_steps'] * (self.params['max_steps'] + 2),
            True
        )
        if create_data:
            self.create_dataset('data/pretrain_rddpg_6', self.signal_gen)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.005,
            decay_steps=180,
            decay_rate=0.95
        )
        self.pretrain_actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate = lr_schedule
        )
        self.action_mse = tf.keras.losses.MeanSquaredError()
        self.omega_mse = tf.keras.losses.MeanSquaredError()
        self.mu_mse = tf.keras.losses.MeanSquaredError()
        self.Z_mse = tf.keras.losses.MeanSquaredError()
        data = random.sample(self.signal_gen.data, 35)
        self.desired_motion = [
            np.expand_dims(x, 0) for y, x, f in data
        ]


    def create_dataset(self, path, signal_gen):
        self.actor.create_data_v2(path, signal_gen, self.env)

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


    def _test_pretrain_actor(self, x, y, W = [1,1,1,1]):
        out, Z, state = self.actor.model(x)
        loss_action = self.action_mse(y[0], out)
        #loss_omega = self.omega_mse(y[1], omega)
        #loss_mu = self.mu_mse(y[2], mu)
        #loss_Z = self.Z_mse(y[3], Z)
        loss = W[0] * loss_action #+ \
            #W[1] * loss_omega + \
            #W[2] * loss_mu + \
            #W[3] * loss_Z
        loss_omega = tf.convert_to_tensor(0.0, dtype = tf.dtypes.float32)
        loss_mu = tf.convert_to_tensor(0.0, dtype = tf.dtypes.float32)
        loss_Z = tf.convert_to_tensor(0.0, dtype = tf.dtypes.float32)
        return loss, [loss_action, loss_omega, loss_mu, loss_Z]

    def _pretrain_actor_v2(self, x, y, W = [1,1,1,1]):
        with tf.GradientTape(persistent=False) as tape:
            actions, Z, state = self.actor.model(x)
            loss_action = self.action_mse(y[0], actions)
            #loss_omega = self.omega_mse(y[1], omega)
            #loss_mu = self.mu_mse(y[2], mu)
            #loss_Z = self.Z_mse(y[3], Z)
            loss = W[0] * loss_action #+ \
                #W[1] * loss_omega + \
                #W[2] * loss_mu + \
                #W[3] * loss_Z
        loss_omega = tf.convert_to_tensor(0.0, dtype = tf.dtypes.float32)
        loss_mu = tf.convert_to_tensor(0.0, dtype = tf.dtypes.float32)
        loss_Z = tf.convert_to_tensor(0.0, dtype = tf.dtypes.float32)
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
        return loss, [loss_action, loss_omega, loss_mu, loss_Z]


    def _pretrain_actor(self, x, y, W = [1,1,1,1]):
        with tf.GradientTape(persistent=False) as tape:
            out, omega, mu, Z = self.actor.model(x)
            loss_action = self.action_mse(y[0], out)
            loss_omega = self.omega_mse(y[1], omega)
            loss_mu = self.mu_mse(y[2], mu)
            loss_Z = self.Z_mse(y[3], Z)
            loss = W[0] * loss_action + \
                W[1] * loss_omega + \
                W[2] * loss_mu + \
                W[3] * loss_Z

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
        return loss, [loss_action, loss_omega, loss_mu, loss_Z]

    def create_init_osc_state(self):
        r = np.ones((self.params['units_osc'],), dtype = np.float32)
        phi = np.zeros((self.params['units_osc'],), dtype = np.float32)
        z = r * np.exp(1j * phi)
        x = np.real(z)
        y = np.imag(z)
        return np.concatenate([x, y], -1)

    def test_actor(self, path, ep = None):
        print('[Actor] Starting Actor Test')
        data = random.sample(self.signal_gen.data, 10)
        y = [np.expand_dims(d[0], 0) for d in data]
        x = [np.expand_dims(d[1], 0) for d in data]
        f = [np.array([[d[2]]], dtype = np.float32) \
            for d in data]
        y = np.concatenate(y, 0)
        x = np.concatenate(x, 0)
        f = np.concatenate(f, 0)
        z = np.concatenate(
            [np.expand_dims(
                self.create_init_osc_state(), 0
        ) for i in range(10)], 0)
        robot_state = np.zeros(
            (10, self.params['robot_state_size']),
            dtype = np.float32
        )
        y_pred = []
        state = tf.repeat(self.actor.gru_recurrent_state_init, 10, 0)
        print('[Actor] Generating Actions')
        steps = self.params['rnn_steps'] * self.params['max_steps'] + 1
        for i in tqdm(range(
            steps
        )):
            inp = [x, robot_state, z, state]
            actions, _, _, z, state = self.actor.model.layers[-1].rnn_cell(inp)
            ac = actions[0].numpy()
            self.env.quadruped.all_legs.move(ac)
            actions = actions.numpy() * np.pi / 3
            y_pred.append(np.expand_dims(
                actions, 1
            ))
        y = y * np.pi / 180.0
        y_pred = np.concatenate(y_pred, 1)
        fig, ax = plt.subplots(4,1, figsize = (5,20))
        for i in range(4):
            ax[i].plot(y_pred[0][:,3*i], 'b', label = 'ankle')
            ax[i].plot(y_pred[0][:,3*i + 1], 'g', label = 'knee')
            ax[i].plot(y_pred[0][:,3*i + 2], 'r', label = 'hip')
            ax[i].legend()
        if ep is not None:
            fig.savefig(os.path.join(
                path,'y_pred_{ep}.png'.format(
                    ep = ep
                )
            ))
        else:
            fig.savefig(os.path.join(path,'y_pred.png'))
        fig, ax = plt.subplots(4,1, figsize = (5,20))
        for i in range(4):
            ax[i].plot(y[0][:steps,3*i], 'b', label = 'ankle real')
            ax[i].plot(y[0][:steps,3*i + 1], 'g', label = 'knee real')
            ax[i].plot(y[0][:steps,3*i + 2], 'r', label = 'hip real')
            ax[i].legend()
        if ep is not None:
            fig.savefig(os.path.join(
                path,'y_{ep}.png'.format(
                    ep = ep
                )
            ))
        else:
            fig.savefig(os.path.join(path,'y.png'))
        plt.close('all')
        print('[Actor] Finishing Actor Test')

    def _pretrain_loop(
        self,
        grad_update,
        test_actor,
        experiment,
        checkpoint_dir,
        name,
        train_dataset,
        test_dataset,
        epochs = None,
        start = 0,
        W = [1.0, 1.0, 1.0, 1.0],
        delta_W = None
    ):
        if epochs is None:
            epochs = self.params['train_episode_count']
        self.test_actor(os.path.join(
            checkpoint_dir,
            'exp{exp}'.format(
                exp = experiment,
            ),
            name
        ))
        total_loss = 0.0
        avg_loss = 0.0
        prev_loss = 1e20
        history_loss = []
        history_loss_action = []
        history_loss_omega = []
        history_loss_mu = []
        history_loss_Z = []
        test_history_loss = []
        test_history_loss_action = []
        test_history_loss_omega = []
        test_history_loss_mu = []
        test_history_loss_Z = []
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

            pkl = open(os.path.join(path, 'loss_omega_{ex}_{name}_{ep}.pickle'.format(
                name = name,
                ex = experiment,
                ep = start - 1
            )), 'rb')
            history_loss_omega = pickle.load(pkl)
            pkl.close()

            pkl = open(os.path.join(path, 'loss_mu_{ex}_{name}_{ep}.pickle'.format(
                name = name,
                ex = experiment,
                ep = start - 1
            )), 'rb')
            history_loss_mu = pickle.load(pkl)
            pkl.close()

            pkl = open(os.path.join(path, 'loss_Z_{ex}_{name}_{ep}.pickle'.format(
                name = name,
                ex = experiment,
                ep = start - 1
            )), 'rb')
            history_loss_Z = pickle.load(pkl)
            pkl.close()
        """
        test_dataset = self.actor.create_pretrain_dataset(
            'data/pretrain_rddpg_6',
            self.params,
            False
        )
        """
        print('[Actor] Starting Actor Pretraining')
        for episode in range(start, epochs):
            print('[Actor] Starting Episode {ep}'.format(ep = episode))
            total_loss = 0.0
            total_loss_action = 0.0
            total_loss_omega = 0.0
            total_loss_mu = 0.0
            total_loss_Z = 0.0
            start = time.time()
            num = 0
            for step, (x, y) in enumerate(train_dataset):
                loss, [loss_action, \
                        loss_omega, \
                        loss_mu, \
                        loss_Z] = grad_update(x, y, W)
                loss = loss.numpy()
                loss_action = loss_action.numpy()
                loss_omega = loss_omega.numpy()
                loss_mu = loss_mu.numpy()
                loss_Z = loss_Z.numpy()
                print('[Actor] Episode {ep} Step {st} Loss: {loss:.5f}'.format(
                    ep = episode,
                    st = step,
                    loss = loss
                ))
                total_loss += loss
                total_loss_action += loss_action
                total_loss_omega += loss_omega
                total_loss_mu += loss_mu
                total_loss_Z += loss_Z
                num += 1
                if step > 100:
                    break
            end = time.time()
            avg_loss = total_loss / num
            avg_loss_action = total_loss_action / num
            avg_loss_omega = total_loss_omega / num
            avg_loss_mu = total_loss_mu / num
            avg_loss_Z = total_loss_Z / num
            print('-------------------------------------------------')
            print('[Actor] Episode {ep} Average Loss: {l:.5f}'.format(
                ep = episode,
                l = avg_loss
            ))
            print('[Actor] Learning Rate: {lr:.5f}'.format(
                lr = self.pretrain_actor_optimizer.lr((episode + 1) * 5))
            )
            print('[Actor] Epoch Time: {time:.5f}s'.format(time = end - start))
            print('-------------------------------------------------')
            history_loss.append(avg_loss)
            history_loss_action.append(avg_loss_action)
            history_loss_omega.append(avg_loss_omega)
            history_loss_mu.append(avg_loss_mu)
            history_loss_Z.append(avg_loss_Z)
            total_loss = 0.0
            total_loss_action = 0.0
            total_loss_omega = 0.0
            total_loss_mu = 0.0
            total_loss_Z = 0.0
            start = time.time()
            num = 0
            for step, (x, y) in enumerate(test_dataset):
                loss, [loss_action, \
                    loss_omega, \
                    loss_mu, \
                    loss_Z] = test_actor(x, y, W)
                loss = loss.numpy()
                loss_action = loss_action.numpy()
                loss_omega = loss_omega.numpy()
                loss_mu = loss_mu.numpy()
                loss_Z = loss_Z.numpy()
                print('[Actor] Test Episode {ep} Step {st} Loss: {loss:.5f}'.format(
                    ep = episode,
                    st = step,
                    loss = loss
                ))
                total_loss += loss
                total_loss_action += loss_action
                total_loss_omega += loss_omega
                total_loss_mu += loss_mu
                total_loss_Z += loss_Z
                num += 1
            end = time.time()
            avg_loss = total_loss / num
            avg_loss_action = total_loss_action / num
            avg_loss_omega = total_loss_omega / num
            avg_loss_mu = total_loss_mu / num
            avg_loss_Z = total_loss_Z / num
            test_history_loss.append(avg_loss)
            test_history_loss_action.append(avg_loss_action)
            test_history_loss_omega.append(avg_loss_omega)
            test_history_loss_mu.append(avg_loss_mu)
            test_history_loss_Z.append(avg_loss_Z)
            print('-------------------------------------------------')
            print('[Actor] Episode {ep} Average Loss: {l:.5f}'.format(
                ep = episode,
                l = avg_loss
            ))
            print('[Actor] Test Time: {time:.5f}s'.format(time = end - start))
            print('-------------------------------------------------')
            if delta_W is not None:
                W = [
                    w * delta_w for w, delta_w in zip(
                        W, delta_W
                    )
                ]
            if episode % self.params['pretrain_test_interval'] == 0:
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
                ax1.set_xlabel('steps')
                ax1.set_ylabel('loss')
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
                ax2.set_xlabel('steps')
                ax2.set_ylabel('loss')
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

                pkl = open(os.path.join(path, 'loss_omega_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(history_loss_omega, pkl)
                pkl.close()
                fig5, ax5 = plt.subplots(1, 1, figsize = (5, 5))
                ax5.plot(history_loss_omega)
                ax5.set_xlabel('steps')
                ax5.set_ylabel('loss')
                ax5.set_title('Total Omega Loss')
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

                pkl = open(os.path.join(path, 'loss_mu_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(history_loss_mu, pkl)
                pkl.close()
                fig5, ax5 = plt.subplots(1, 1, figsize = (5, 5))
                ax5.plot(history_loss_mu)
                ax5.set_xlabel('steps')
                ax5.set_ylabel('loss')
                ax5.set_title('Total mu Loss')
                fig5.savefig(
                    os.path.join(
                        path,
                        'loss_mu_{ex}_{name}_{ep}.png'.format(
                            name = name,
                            ex = experiment,
                            ep = episode
                        )
                    )
                )

                pkl = open(os.path.join(path, 'loss_Z_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(history_loss_Z, pkl)
                pkl.close()
                fig5, ax5 = plt.subplots(1, 1, figsize = (5, 5))
                ax5.plot(history_loss_Z)
                ax5.set_xlabel('steps')
                ax5.set_ylabel('loss')
                ax5.set_title('Total Z  Loss')
                fig5.savefig(
                    os.path.join(
                        path,
                        'loss_Z_{ex}_{name}_{ep}.png'.format(
                            name = name,
                            ex = experiment,
                            ep = episode
                        )
                    )
                )

                pkl = open(os.path.join(path, 'test_loss_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(test_history_loss, pkl)
                pkl.close()
                fig1, ax1 = plt.subplots(1, 1, figsize = (5, 5))
                ax1.plot(test_history_loss)
                ax1.set_xlabel('steps')
                ax1.set_ylabel('loss')
                ax1.set_title('Total Loss')
                fig1.savefig(
                    os.path.join(
                        path,
                        'test_loss_{ex}_{name}_{ep}.png'.format(
                            name = name,
                            ex = experiment,
                            ep = episode
                        )
                    )
                )

                pkl = open(os.path.join(path, 'test_loss_action_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(test_history_loss_action, pkl)
                pkl.close()
                fig2, ax2 = plt.subplots(1, 1, figsize = (5, 5))
                ax2.plot(test_history_loss_action)
                ax2.set_xlabel('steps')
                ax2.set_ylabel('loss')
                ax2.set_title('Total Action Loss')
                fig2.savefig(
                    os.path.join(
                        path,
                        'test_loss_action_{ex}_{name}_{ep}.png'.format(
                            name = name,
                            ex = experiment,
                            ep = episode
                        )
                    )
                )

                pkl = open(os.path.join(path, 'test_loss_omega_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(test_history_loss_omega, pkl)
                pkl.close()
                fig5, ax5 = plt.subplots(1, 1, figsize = (5, 5))
                ax5.plot(test_history_loss_omega)
                ax5.set_xlabel('steps')
                ax5.set_ylabel('loss')
                ax5.set_title('Total Omega Loss')
                fig5.savefig(
                    os.path.join(
                        path,
                        'test_loss_omega_{ex}_{name}_{ep}.png'.format(
                            name = name,
                            ex = experiment,
                            ep = episode
                        )
                    )
                )

                pkl = open(os.path.join(path, 'test_loss_mu_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(test_history_loss_mu, pkl)
                pkl.close()
                fig5, ax5 = plt.subplots(1, 1, figsize = (5, 5))
                ax5.plot(test_history_loss_mu)
                ax5.set_xlabel('steps')
                ax5.set_ylabel('loss')
                ax5.set_title('Total mu Loss')
                fig5.savefig(
                    os.path.join(
                        path,
                        'test_loss_mu_{ex}_{name}_{ep}.png'.format(
                            name = name,
                            ex = experiment,
                            ep = episode
                        )
                    )
                )

                pkl = open(os.path.join(path, 'test_loss_Z_{ex}_{name}_{ep}.pickle'.format(
                    name = name,
                    ex = experiment,
                    ep = episode
                )), 'wb')
                pickle.dump(test_history_loss_Z, pkl)
                pkl.close()
                fig5, ax5 = plt.subplots(1, 1, figsize = (5, 5))
                ax5.plot(test_history_loss_Z)
                ax5.set_xlabel('steps')
                ax5.set_ylabel('loss')
                ax5.set_title('Total Z  Loss')
                fig5.savefig(
                    os.path.join(
                        path,
                        'test_loss_Z_{ex}_{name}_{ep}.png'.format(
                            name = name,
                            ex = experiment,
                            ep = episode
                        )
                    )
                )

                plt.close('all')

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
            'exp{exp}'.format(
                exp = experiment,
            ),
            name
        ))

    def pretrain_actor(self, experiment, \
            checkpoint_dir = 'weights/actor_pretrain', \
            name = 'pretrain_actor'):
        path = os.path.join(checkpoint_dir, 'exp{exp}'.format(exp = experiment))
        """
        model = self.actor.create_pretrain_actor_cell(
            self.params
        )
        print(model.summary())
        self.actor.set_model(
            model
        )
        """
        print(self.actor.model.summary())
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(
            os.path.join(
                path, name
            )
        ):
            os.mkdir(os.path.join(
                path, name
            ))
        train_dataset, test_dataset = self.actor.create_pretrain_dataset(
            'data/pretrain_rddpg_6',
            self.params
        )
        self._pretrain_loop(
            self._pretrain_actor_v2, \
            self._test_pretrain_actor, experiment, checkpoint_dir, name,
            train_dataset = train_dataset,
            test_dataset = test_dataset,
            W = [1.0, 1.0, 1.0, 0.1],
            delta_W = [1.0, 1.0, 0.0, 0.0]
        )

    def _add_noise(self, action):
        # noise theta and sigma scaled by 0.1 for exp5
        self._noise = max(self.epsilon, 0) * self.OU.function(
            action,
            0.0,
            0.15,
            0.2
        )
        action = action + self._noise
        return action

    def learn(self, model_dir, experiment, start_epoch = 0, per = False, \
            her = False):
        if per:
            print('[RDDPG] Initializing PER')
            self.replay_buffer = PER(self.params)
            raise NotImplementedError
        if her:
            print('[RDDPG] Initializing HER')
            raise NotImplementedError
        print('[RDDPG] Training Start')
        hist_critic_loss = []
        hist_actor_loss = []
        hist_rewards = []
        total_rewards = []
        avg_reward = []
        COT = []
        stability = []
        d1 = []
        d2 = []
        d3 = []
        motion = []
        print('[RDDPG] Starting Pretraining Test')
        total_reward = 0.0
        step = 0
        tot_loss = 0.0
        break_loop = False
        self.epsilon -= 1/self.params['EXPLORE']
        self.test_actor(model_dir)
        done = False
        self.current_time_step = self.env.reset()
        self._state = self.current_time_step.observation
        self._gru_state = self.actor.gru_recurrent_state_init
        while(not done):
            penalty = tf.convert_to_tensor(0.0, dtype = tf.dtypes.float32)
            start = time.perf_counter()
            action, _, _, Z, state = \
                self.actor.model.layers[-1].rnn_cell(
                    self._state + [self._gru_state]
                )
            self._action = self._add_noise(action)
            self._gru_state = state
            self._osc_state = Z
            if math.isnan(np.sum(self._action.numpy())):
                print('[RDDPG] Action value NaN. Ending Episode')
                penalty += tf.convert_to_tensor(-5.0, dtype = tf.dtypes.float32)
                self._action = tf.zeros_like(self._action)
            try:
                last_step = False
                first_step = False
                if step == 0:
                    first_step = True
                if step < 19:#-1 + self.params['max_steps'] * self.params['rnn_steps']:
                    last_step = False
                else:
                    last_step = True
                    done = True
                self.current_time_step = self.env.step(
                    [self._action, self._osc_state],
                    self._state[0][0].numpy(),
                    last_step = last_step,
                    first_step = first_step,
                    version = 2
                )
            except FloatingPointError:
                print('[RDDPG] Floating Point Error in reward computation')
                penalty += tf.convert_to_tensor(-1.0, dtype = tf.dtypes.float32)
            except Exception as e:
                raise e
            reward = self.current_time_step.reward + penalty

            self._state = self.current_time_step.observation
            print('[RDDPG] Step {step} Reward {reward:.5f} Time {time:.5f}'.format(
                step = step,
                reward = reward.numpy(),
                time = time.perf_counter() - start
            ))
            if self.current_time_step.step_type == \
                tfa.trajectories.time_step.StepType.LAST:
                done = True
            if step > 19:#self.params['max_steps'] * self.params['rnn_steps']:
                done = True
            step += 1
        self.current_time_step = self.env.reset()
        self._state = self.current_time_step.observation
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
                'total_rewards_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            total_rewards = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'hist_critic_loss_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            hist_critic_loss = pickle.load(pkl)
            pkl.close()
            pkl = open(os.path.join(
                model_dir,
                'hist_actor_loss_ep{ep}.pickle'.format(
                    ep = ep,
                )
            ), 'rb')
            hist_actor_loss = pickle.load(pkl)
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
        while ep < self.params['train_episode_count']:
            goal_id = np.random.randint(0, len(self.desired_motion))
            desired_motion = self.desired_motion[goal_id]
            enc_goals.append(desired_motion[0])
            self._action = self.env._action_init
            self._gru_state = self.actor.gru_recurrent_state_init
            self._noise = self._noise_init
            self.env.set_motion_state(desired_motion[0])
            self.current_time_step = self.env.reset()
            self._state = self.current_time_step.observation
            self._osc_state = self._state[-1]
            print('[RDDPG] Starting Episode {i}'.format(i = ep))
            total_reward = 0.0
            step = 0
            tot_loss = 0.0
            break_loop = False
            self.epsilon -= 1/self.params['EXPLORE']
            start = None
            observations = []
            actions = []
            rewards = []
            next_states = []
            next_states.append(self._state)
            recurrent_state_init = [
                self._state[-1],
                self.actor.gru_recurrent_state_init
            ]
            done = False
            while(not done):
                penalty = tf.convert_to_tensor(0.0, dtype = tf.dtypes.float32)
                start = time.perf_counter()
                action, _, _, Z, state = \
                    self.actor.model.layers[-1].rnn_cell(
                        self._state + [self._gru_state]
                    )
                self._action = self._add_noise(action)
                self._gru_state = state
                self._osc_state = Z
                if math.isnan(np.sum(self._action.numpy())):
                    print('[RDDPG] Action value NaN. Ending Episode')
                    penalty += tf.convert_to_tensor(-5.0, dtype = tf.dtypes.float32)
                    self._action = tf.zeros_like(self._action)
                try:
                    last_step = False
                    first_step = False
                    if step == 0:
                        first_step = True
                    if step < self.params['max_steps'] * self.params['rnn_steps'] - 1:
                        last_step = False
                    else:
                        last_step = True
                        done = True
                    self.current_time_step = self.env.step(
                        [self._action, self._osc_state],
                        self._state[0][0].numpy(),
                        last_step = last_step,
                        first_step = first_step,
                        version = 2
                    )
                except FloatingPointError:
                    print('[RDDPG] Floating Point Error in reward computation')
                    penalty += tf.convert_to_tensor(-5.0, dtype = tf.dtypes.float32)
                reward = self.current_time_step.reward + penalty
                motion.append(self.env.quadruped.r_motion)
                COT.append(self.env.quadruped.COT)
                stability.append(self.env.quadruped.stability)
                d1.append(self.env.quadruped.d1)
                d2.append(self.env.quadruped.d2)
                d3.append(self.env.quadruped.d3)
                hist_rewards.append(reward.numpy())

                rewards.append(reward)
                observations.append(self._state)
                self._state = self.current_time_step.observation
                next_states.append(self._state)
                actions.append(self._action)
                total_reward += reward.numpy()
                print('[RDDPG] Episode {ep} Step {step} Reward {reward:.5f} Time {time:.5f}'.format(
                    ep = ep,
                    step = step,
                    reward = reward.numpy(),
                    time = time.perf_counter() - start
                ))
                if self.current_time_step.step_type == \
                    tfa.trajectories.time_step.StepType.LAST:
                    done = True
                if step > self.params['max_steps'] * self.params['rnn_steps'] - 1:
                    done = True
                step += 1
            experience = [
                observations,
                rewards,
                actions,
                next_states,
                recurrent_state_init
            ]
            self.replay_buffer.add_batch(experience)
            start = time.perf_counter()
            states, rewards, actions, next_states, \
                recurrent_state_init, batch_size = \
                self.get_batch()
            target_actions, Z, state = self.actor.target_model([
                next_states + recurrent_state_init
            ])
            recurrent_state = [tf.repeat(
                rci,
                batch_size,
                0
            ) for rci in self.critic.recurrent_state_init]
            inputs = next_states + [target_actions]
            target_q_values = self.critic.target_model(inputs)
            y = tf.concat([
                rewards[:, :-1] + \
                    self.params['GAMMA'] * target_q_values[:, 1:-1],
                rewards[:, -1:]
            ], 1)
            loss = self.critic.train(states, actions, y)
            hist_critic_loss.append(loss.numpy())
            actor_loss = self.train_actor(
                states, recurrent_state_init, \
                actions
            )
            hist_actor_loss.append(actor_loss.numpy())
            self.actor.target_train()
            self.critic.target_train()
            print('[RDDPG] Critic Loss {loss} Actor Loss {ac_loss}'.format(
                loss = loss.numpy(),
                ac_loss = actor_loss.numpy()
            ))
            print('[RDDPG] Total Reward {reward:.5f} Time {time:.5f}'.format(
                reward = total_reward,
                time = time.perf_counter() - start
            ))
            if ep % self.params['TEST_AFTER_N_EPISODES'] == 0:
                self.save(model_dir, ep, hist_rewards, total_rewards, \
                    hist_critic_loss, hist_actor_loss, COT, motion, \
                    stability, d1, d2, d3)
            total_rewards.append(total_reward)
            ep += 1
            print('[DDPG] Starting Next Episode')

    def train_actor(self, \
        states, recurrent_state_init, \
        actions
    ):
        with tf.GradientTape() as tape:
            a_for_grad, Z, state = self.actor.model([
                states + recurrent_state_init
            ])
            inputs = states + [a_for_grad]
            q_val = self.critic.model(inputs)
            actor_loss = -tf.math.reduce_mean(q_val)
        actor_grads = tape.gradient(
            actor_loss,
            self.actor.model.trainable_variables
        )
        self.actor.optimizer.apply_gradients(
            zip(
                actor_grads,
                self.actor.model.trainable_variables
            )
        )
        return actor_loss

    def save(self, model_dir, ep, rewards, total_reward, hist_critic_loss, \
            hist_actor_loss, COT, motion, stability, d1, d2, d3, \
            tree = None, enc_goals = None):
        print('[RDDPG] Saving Data')
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
            print('[RDDPG] Saving PER priorities')
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
            print('[RDDPG] Saving HER goals')
            path = os.path.join(
                model_dir,
                'enc_goals.pickle'
            )
            pkl = open(path, 'wb')
            pickle.dump(enc_goals, pkl)
            pkl.close()
        print('[RDDPG] Saving Model')
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
            'hist_critic_loss_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(hist_critic_loss, pkl)
        pkl.close()
        fig3, ax3 = plt.subplots(1,1,figsize = (5,5))
        ax3.plot(hist_critic_loss)
        ax3.set_ylabel('critic loss')
        ax3.set_xlabel('steps')
        fig3.savefig(os.path.join(
            model_dir,
            'hist_critic_loss_ep{ep}.png'.format(
                ep = ep,
            )
        ))

        pkl = open(os.path.join(
            model_dir,
            'hist_actor_loss_ep{ep}.pickle'.format(
                ep = ep,
            )
        ), 'wb')
        pickle.dump(hist_actor_loss, pkl)
        pkl.close()
        fig3, ax3 = plt.subplots(1,1,figsize = (5,5))
        ax3.plot(hist_actor_loss)
        ax3.set_ylabel('critic loss')
        ax3.set_xlabel('steps')
        fig3.savefig(os.path.join(
            model_dir,
            'hist_actor_loss_ep{ep}.png'.format(
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

    def get_batch(self):
        batch = self.replay_buffer.get_next(
            self.params['BATCH_SIZE']
        )
        batch_size = len(batch)
        states = [[] for i in range(
            len(
                self.params['observation_spec']
            ) - 1
        )]
        rewards = []
        actions = []
        next_states = [[] for i in range(
            len(
                self.params['observation_spec']
            ) - 1
        )]
        _recurrent_state_init = [[] for i in range(2)]
        step_types = []
        for item in batch:
            state = [[] for i in range(len(states))]
            next_state = [[] for i in range(len(next_states))]
            step_type = []
            for j, st in enumerate(item[0]):
                for k, s in enumerate(st[:-1]):
                    state[k].append(s)
            state = [tf.expand_dims(tf.concat(s, 0), 0) for s in state]
            reward = tf.expand_dims(tf.concat([tf.expand_dims(
                tf.expand_dims(
                    rw, 0
                ), 0
            ) for rw in item[1]], 0), 0)
            action = tf.expand_dims(tf.concat(item[2], 0), 0)
            for j, st in enumerate(item[3]):
                for k, s in enumerate(st[:-1]):
                    next_state[k].append(s)
            next_state = [tf.expand_dims(tf.concat(s, 0), 0) \
                for s in next_state]

            for i in range(len(states)):
                states[i].append(state[i])
            rewards.append(reward)
            actions.append(action)
            for i in range(len(next_states)):
                next_states[i].append(next_state[i])
            for j, rc in enumerate(item[4]):
                _recurrent_state_init[j].append(rc)

        states = [tf.concat(state, 0) for state in states]
        rewards = tf.concat(rewards, 0)
        actions = tf.concat(actions, 0)
        next_states = [tf.concat(state, 0) for state in next_states]
        _recurrent_state_init = [tf.concat(ars, 0) \
            for ars in _recurrent_state_init]

        return states, rewards, actions, next_states, \
            _recurrent_state_init, batch_size

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
    learner = Learner(params, args.experiment, False)
    """
    learner.pretrain_actor(
        args.experiment,
        args.out_path,
        'pretrain_actor'
    )
    """
    print(learner.actor.model.summary())
    """
    learner.load_actor('weights/actor_pretrain/exp53/pretrain_actor/actor_pretrained_pretrain_actor_53_5.ckpt',
            'weights/actor_pretrain/exp53/pretrain_actor/actor_pretrained_pretrain_actor_53_5.ckpt')
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
    #"""
