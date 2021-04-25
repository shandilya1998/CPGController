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

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class Learner:
    def __init__(self, params, experiment, create_data = False):
        self.params = params
        self.experiment = experiment
        self.actor = ActorNetwork(self.params)
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
            self.create_dataset('../input/rddpgpretraindata5', self.signal_gen)
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

    def create_dataset(self, path, signal_gen):
        self.actor.create_data(path, signal_gen, self.env)

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
        out, omega, mu, Z = self.actor.model(x)
        loss_action = self.action_mse(y[0], out)
        loss_omega = self.omega_mse(y[1], omega)
        loss_mu = self.mu_mse(y[2], mu)
        loss_Z = self.Z_mse(y[3], Z)
        loss = W[0] * loss_action + \
            W[1] * loss_omega + \
            W[2] * loss_mu + \
            W[3] * loss_Z
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

    def test_actor(self, path, ep = None):
        print('[Actor] Starting Actor Test')
        data = random.sample(self.signal_gen.data, self.params['pretrain_bs'])
        y = [np.d[0] for d in data]
        x = [np.expand_dims(d[1]) for d in data]
        f = [np.array([[d[2]]], dtype = np.float32) \
            for d in data]
        y = np.concatenate(y, 0)
        x = np.concatenate(x, 0)
        f = np.concatenate(f, 0)
        z = np.concatenate(
            [np.expand_dims(
                self.env.quadruped.create_init_osc_state(), 0
        ) for i in range(self.params['pretrain_bs'])], 0)
        mod_state = np.zeros(
            (self.params['pretrain_bs'], 2 * self.params['units_osc']),
            dtype = np.float32
        )
        y_pred = []
        for i in range(
            self.params['max_steps'] * self.params['rnn_steps']
        ):
            inp = [x, mod_state, z]
            actions, _, _, z = self.actor.model(inp)
            actions = actions.numpy() * np.pi / 3
            y_pred.append(actions)
        y = y * np.pi / 3
        y_pred = np.concatenate(y_pred, 0)
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
            ax[i].plot(y[0][:,3*i], 'b', label = 'ankle real')
            ax[i].plot(y[0][:,3*i + 1], 'g', label = 'knee real')
            ax[i].plot(y[0][:,3*i + 2], 'r', label = 'hip real')
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
        W = [1.0, 1.0, 1.0, 1.0]
    ):
        if epochs is None:
            epochs = self.params['train_episode_count']
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
            'data/pretrain_rddpg_4',
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
        model = self.actor.create_pretrain_actor_cell(
            self.params
        )
        print(model.summary())
        self.actor.set_model(
            model
        )
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(
            os.path.join(
                path, 'pretrain_enc'
            )
        ):
            os.mkdir(os.path.join(
                path, 'pretrain_enc'
            ))
        if not os.path.exists(
            os.path.join(
                path, name
            )
        ):
            os.mkdir(os.path.join(
                path, name
            ))
        train_dataset, test_dataset = self.actor.create_pretrain_dataset(
            'data/pretrain_rddpg_4',
            self.params
        )
        
        
        self._pretrain_loop(
            self._pretrain_actor, \
            self._test_pretrain_actor, experiment, checkpoint_dir, 'pretrain_enc',
            train_dataset = train_dataset,
            test_dataset = test_dataset,
            W = [0.1, 1.0, 0.0, 1.0]
        )
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.005,
            decay_steps=180,
            decay_rate=0.95
        )
        self.pretrain_actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate = lr_schedule
        )
        
        self._pretrain_loop(
            self._pretrain_actor, \
            self._test_pretrain_actor, experiment, checkpoint_dir, name,
            train_dataset = train_dataset,
            test_dataset = test_dataset,
            W = [1.0, 0.1, 0.0, 0.1]
        )

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
    learner.pretrain_actor(
        args.experiment,
        args.out_path,
        'pretrain_actor'
    )
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
