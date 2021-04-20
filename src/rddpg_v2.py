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
        self.dt = self.params['dt']
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


    def _pretrain_actor(self, x, y, W = [1,1]):
        with tf.GradientTape(persistent=False) as tape:
            out, omega, _ = self.actor.model(x)
            loss_action = self.action_mse(y[0], out)
            loss_omega = self.omega_mse(y[1], omega)
            loss = W[0] * loss_action + W[1] * loss_omega

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
        return loss, [loss_action, loss_omega]

    def test_actor(self, path, ep = None):
        print('[Actor] Starting Actor Test')
        step, (x, y) = next(enumerate(
            self.actor.create_pretrain_dataset(
                'data/pretrain_rddpg_2',
                self.params
            )
        ))
        actions, omega, _ = self.actor.model(x)
        bs = actions.shape[0]
        action_dim = actions.shape[-1]
        steps = actions.shape[2]
        max_steps = actions.shape[1]
        shape = (bs, steps * max_steps, action_dim)
        actions = tf.reshape(actions, shape) * np.pi / 3
        bs = y[0].shape[0]
        action_dim = y[0].shape[-1]
        steps = y[0].shape[2]
        max_steps = y[0].shape[1]
        shape = (bs, steps * max_steps, action_dim)
        y = tf.reshape(y[0], shape) * np.pi / 3
        fig, ax = plt.subplots(4,1, figsize = (5,20))
        for i in range(4):
            ax[i].plot(actions[0][:,3*i], 'b', label = 'ankle')
            ax[i].plot(actions[0][:,3*i + 1], 'g', label = 'knee')
            ax[i].plot(actions[0][:,3*i + 2], 'r', label = 'hip')
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
        experiment,
        checkpoint_dir,
        name,
        epochs = None,
        start = 0,
        W = [1.0, 1.0]
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

            pkl = open(os.path.join(path, 'loss_omega_{ex}_{name}_{ep}.pickle'.format(
                name = name,
                ex = experiment,
                ep = start - 1
            )), 'rb')
            history_loss_omega = pickle.load(pkl)
            pkl.close()
        dataset = self.actor.create_pretrain_dataset(
            'data/pretrain_rddpg_2',
            self.params
        )
        print('[Actor] Dataset {ds}'.format(ds = dataset))
        print('[Actor] Starting Actor Pretraining')
        for episode in range(start, epochs):
            print('[Actor] Starting Episode {ep}'.format(ep = episode))
            total_loss = 0.0
            total_loss_action = 0.0
            total_loss_omega = 0.0
            start = time.time()
            num = 0
            for step, (x, y) in enumerate(dataset):
                loss, [loss_action, loss_omega] = grad_update(x, y, W)
                loss = loss.numpy()
                loss_action = loss_action.numpy()
                loss_omega = loss_omega.numpy()
                print('[Actor] Episode {ep} Step {st} Loss: {loss}'.format(
                    ep = episode,
                    st = step,
                    loss = loss
                ))
                total_loss += loss
                total_loss_action += loss_action
                total_loss_omega += loss_omega
                num += 1
                if step > 100:
                    break
            end = time.time()
            avg_loss = total_loss / num
            avg_loss_action = total_loss_action / num
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
                'exp{exp}'.format(
                    exp = experiment,
                ),
                name
            ), episode)

    def pretrain_actor(self, experiment, \
            checkpoint_dir = 'weights/actor_pretrain', \
            name = 'pretrain_actor'):
        path = os.path.join(checkpoint_dir, 'exp{exp}'.format(exp = experiment))
        self.actor.set_model(
            self.actor.create_pretrain_actor_network(
                self.params
            )
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
        self._pretrain_loop(
            self._pretrain_actor, experiment, checkpoint_dir, 'pretrain_enc',
            W = [0.001, 1.0]
        )
        self._pretrain_loop(
            self._pretrain_actor, experiment, checkpoint_dir, name,
            W = [1.0, 0.001]
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
    learner = Learner(params, args.experiment, False)
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
