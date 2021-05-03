import numpy as np
import torch
from layers.torch_l import Actor, Critic, ActorCell
from rl.torch.memory import SequentialMemory, EpisodicMemory
from rl.torch.random_process import OrnsteinUhlenbeckProcess
from rl.torch.util import *

criterion = torch.nn.MSELoss()

class Agent(object):
    def __init__(self, params, steps = None, cell = None, USE_CUDA = False):
        self.params = params
        self.steps = steps
        if steps is None:
            self.steps = self.params['rnn_steps'] * self.params['max_steps']
        self.trajectory_length = self.steps
        self.nb_actions= self.params['action_dim']
        self.cell = cell

        # Create Actor and Critic Network
        self.actor = Actor(self.params, self.cell)
        self.actor_target = Actor(self.params, self.cell)

        self.critic = Critic(self.params)
        self.critic_target = Critic(self.params)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        #Create replay buffer
        self.random_process = OrnsteinUhlenbeckProcess(
            size=self.nb_actions,
            theta= params['ou_theta'],
            mu=params['ou_mu'],
            sigma=params['ou_sigma']
        )

        # Hyper-parameters
        self.batch_size = self.params['BATCH_SIZE']
        self.tau = self.params['TAU']
        self.discount = self.params['GAMMA']
        self.depsilon = 1.0 / self.params['EXPLORE']

        self.epsilon = self.params['epsilon']
        self.is_training = True

        if USE_CUDA: self.cuda()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def random_action(self):
        action = to_tensor(np.expand_dims(
            np.random.uniform(-np.pi/3, np.pi/3, self.nb_actions), 0
        ))
        return action

    def select_action(self, state, noise_enable=True, decay_epsilon=True):
        action, _ = self.actor(torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0))
        action = to_numpy(action).squeeze(0)
        if noise_enable == True:
            action += self.is_training * max(self.epsilon, 0)*self.random_process.sample()

        action = np.clip(action, -1., 1.)
        if decay_epsilon:
            self.epsilon -= self.depsilon
        return to_tensor(np.expand_dims(action, 0))

    def reset_gru_hidden_state(self, done=True):
        self.actor.reset_gru_hidden_state(done)

    def reset(self):
        self.random_process.reset_states()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def load_weights(self, output):
        if output is None: return False

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

        return True


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
