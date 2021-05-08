from rl.torch.agent import Agent
from rl.torch.constants import params
from learn import SignalDataGen
from rl.torch.env import Env
from rl.torch.evaluator import Evaluator
from rl.torch.memory import EpisodicMemory
import os
import torch
import numpy as np
import pickle
import time
import random

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
DEVICE = 'cpu'
if USE_CUDA:
    DEVICE = 'gpu'

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return torch.autograd.Variable(
        torch.from_numpy(ndarray), requires_grad=requires_grad
    ).type(dtype)

class GaitDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, batch_size):
        Y = np.load(
<<<<<<< HEAD
            os.path.join(data_dir, 'Y.npy'),
            allow_pickle = True,
            fix_imports=True
        )
        X_0 = np.load(
            os.path.join(data_dir, 'X_0.npy'),
            allow_pickle = True,
            fix_imports=True
        )
        X_1 = np.load(
            os.path.join(data_dir, 'X_1.npy'),
            allow_pickle = True,
            fix_imports=True
        )
        F = np.load(
            os.path.join(data_dir, 'F.npy'),
            allow_pickle = True,
            fix_imports=True
        )
        MU = np.load(
            os.path.join(data_dir, 'MU.npy'),
            allow_pickle = True,
            fix_imports=True
=======
            os.path.join(data_dir, 'Y.npy')
        )
        X_0 = np.load(
            os.path.join(data_dir, 'X_0.npy')
        )
        X_1 = np.load(
            os.path.join(data_dir, 'X_1.npy')
        )
        F = np.load(
            os.path.join(data_dir, 'F.npy')
        )
        MU = np.load(
            os.path.join(data_dir, 'MU.npy')
>>>>>>> 51b1d0dec8c48ac9194f5638088efcdc21b46f1b
        )
        X = list(zip(X_0, X_1))
        Y = [Y[i] for i in range(Y.shape[0])]
        self.samples = list(zip(X, Y))
        self.batch_size = batch_size
        self.create_batches()

    def shuffle(self):
        random.shuffle(self.samples)
        self.create_batches()

    def create_batches(self):
        self.batches = []
        num_batches = len(self.samples) // self.batch_size
        for i in range(num_batches):
            x_0 = []
            x_1 = []
            y = []
            for step, (x_, y_) in enumerate(
                self.samples[i * self.batch_size: (i+1)*self.batch_size]
            ):
                x_0.append(np.expand_dims(x_[0], 0))
                x_1.append(np.expand_dims(x_[1], 0))
                y.append(np.expand_dims(y_, 0))
            x_0 = np.concatenate(x_0, 0)
            x_1 = np.concatenate(x_1, 0)
            y = np.concatenate(y, 0)
            self.batches.append(((x_0, x_1), y))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

prev_loss = 1e20
total_loss = []
val_reward = []
ACTIONS = []
OBSERVATIONS = [[], []]
HS = [[], []]
REWARDS = []
DONES = []
STOP = False

def execute_policy(x, y, batch_size, agent, actor_optim, \
        step, epoch, env, memory, checkpoint_dir, \
        validate = False, train = True):
    steps = y.shape[1]
    loss = 0.0
    start = time.perf_counter()
    agent.actor.zero_grad()
    h = torch.autograd.Variable(torch.zeros(batch_size, params['units_robot_state'][0])).type(FLOAT)
    z = torch.autograd.Variable(torch.zeros(batch_size, 2 * params['units_osc'])).type(FLOAT)
    ob = None
    first_step = False
    last_step = False
    ob = env.reset()
    total_reward = 0.0
    actions = []
    observations = []
    hs = []
    rewards = []
    dones = []
    for i in range(steps):
        action, (h, z) = agent.actor(
            to_tensor(x[0][:, i], volatile = True),
            to_tensor(x[1][:, i], volatile = True),
            (h, z)
        )
        loss += torch.nn.functional.mse_loss(
            action,
            to_tensor(y[:, i], volatile = True)
        ) / steps
        if i == 0:
            first_step = True
        else:
            first_step = False
        if i == steps -1:
            last_step = True
        else:
            last_step = False
        env.quadruped.set_motion_state(x[0][0, i])
        ob, reward, done, info = env.step(
            action.detach().cpu(),
            x[0][0, i],
            first_step,
            last_step,
            version = params['step_version']
        )
        total_reward += reward
        val_reward.append(reward)
        memory.append(
            [to_tensor(o) for o in ob],
            torch.squeeze(action, 0),
            [torch.squeeze(h_[:1], 0) for h_ in [h, z]],
            reward,
            done
        )
    if train:
        loss.backward()
        actor_optim.step()
        actor_optim.zero_grad()

    total_loss.append(loss.detach().cpu().numpy())
    print('[RDDPG] Epoch {ep} Step {st} Loss {ls:.5f} Time {t:.5f}'.format(
        ep = epoch,
        st = step,
        ls = loss.detach().cpu().numpy(),
        t = time.perf_counter() - start
    ))
<<<<<<< HEAD
    print('[RDDPG] Epoch {ep} Step {st} Reward {ls:.5f}'.format(
=======
    if validate:
        print('[RDDPG] Epoch {ep} Step {st} Reward {ls:.5f}'.format(
>>>>>>> 51b1d0dec8c48ac9194f5638088efcdc21b46f1b
            ep = epoch,
            st = step,
            ls = total_reward
        ))
<<<<<<< HEAD
    if validate:
=======
>>>>>>> 51b1d0dec8c48ac9194f5638088efcdc21b46f1b
        pkl = open(
            os.path.join(checkpoint_dir, 'loss_{}.pickle'.format(
                epoch
            )), 'wb'
        )
        pickle.dump(total_loss, pkl)
        pkl.close()

        pkl = open(
            os.path.join(checkpoint_dir, 'rewards_{}.pickle'.format(
                epoch
            )), 'wb'
        )
        pickle.dump(val_reward, pkl)
        pkl.close

        pkl = open(
            os.path.join(checkpoint_dir, 'memory.pickle'), 'wb'
        )
        pickle.dump(memory, pkl)
        pkl.close()

def pretrain(epochs, batch_size, checkpoint_dir, experiment, \
        dataset, agent, actor_optim, env, memory):
    loss = []
    reward = 0.0
    validate = False
    step, (x, y) = next(enumerate(dataset))
    steps = y.shape[1]
<<<<<<< HEAD
    prev_loss = 1e20
=======
>>>>>>> 51b1d0dec8c48ac9194f5638088efcdc21b46f1b
    execute_policy(x, y, batch_size, agent, \
        actor_optim, 0, 0, env, memory, checkpoint_dir, \
        False, False)
    for epoch in range(epochs):
        if epoch % 2 == 0:
            validate = True
        else:
            validate = False
        for step, (x, y) in enumerate(dataset):
            execute_policy(x, y, batch_size, agent, \
                actor_optim, step, epoch, env, memory, \
                checkpoint_dir, validate)
        if epoch % 3 == 0:
            if prev_loss < total_loss[-1]:
                break
            else:
                prev_loss = total_loss[-1]
        dataset.shuffle()

if __name__ == '__main__':
    experiment = 57
    batch_size = 15
    dataset = GaitDataset('data/pretrain_rddpg_6', batch_size)
    agent = Agent(params)
    actor_optim  = torch.optim.Adam(
        agent.actor.parameters(),
        lr = params['LRA']
    )
    env = Env(
        params,
        experiment
    )
    step, (x, y) = next(enumerate(dataset))
    steps = y.shape[1]
    memory = EpisodicMemory(
        capacity = params['BUFFER_SIZE'],
        max_episode_length = steps,
        window_length = None
    )
    with torch.autograd.set_detect_anomaly(True):
        pretrain(
            1000,
            batch_size,
            'weights/actor_pretrain/exp57',
            experiment,
            dataset,
            agent,
            actor_optim,
            env,
            memory
        )
