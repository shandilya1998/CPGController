import tensorflow as tf
from collections import deque
import numpy as np
import random
import tf_agents as tfa
from rl.sum_tree import SumTree
import math

class OU(object):
    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)


class Data(tf.keras.utils.Sequence):
    def __init__(self, buff, data_spec, batch_size = None):
        if batch_size == None:
            self.bs = 1 
        self.buffer = buff
        self.data_spec = data_spec
        self.num_experiences = len(self.buffer)
        if self.num_experiences < self.bs:
            raise ValueError('Expected Batch Size to be less than number of experiences(`{n}`), got `{b}`'.format(n = self.num_expriences, b = self.bs))

    def on_epoch_end(self):
        random.shuffle(self.buffer)

    def get_batch(self, index):
        index = index*self.bs
        # Randomly sample batch_size examples
        return self.buffer[index : index + self.bs]

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item

    def __getitem__(self, index):
        batch = self.get_batch(index)
        out = []
        for i in range(self.bs):
            if i == 0:
                out.extend(batch[i])
            else:
                for i, tensor in enumerate(out):
                    out[i] = tf.concat([tensor, batch[i]], axis = 0)
        return out

class ReplayBuffer(tfa.replay_buffers.replay_buffer.ReplayBuffer):

    def __init__(self, params):
        super(ReplayBuffer, self).__init__(
            params['data_spec'],
            params['BUFFER_SIZE']
        )
        self.params = params
        self.num_experiences = 0
        self.buffer = deque()

    def gather_all(self):
        return self.buffer

    def get_next(self, batch_size = 1):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def add_batch(self, experience):
        if self.num_experiences < self.capacity:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def __len__(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def clear(self):
        self.buffer = deque()
        self.num_experiences = 0

    def as_dataset(self, sample_batch_size):
        return tf.data.Dataset.from_generator(
            Data(
                self.buffer, 
                self.data_spec, 
                sample_batch_size
            )
        )

    def set_buffer(self, _buffer):
        self.buffer = _buffer

class PER(ReplayBuffer):
    def __init__(self, params):
        super(PER, self).__init__(params)
        self.epsilon  = params['epsilon']
        self.alpha = params['alpha']
        self.betaInit = params['beta_init']
        self.betaFinal = params['beta_final']
        self.betaFinalAt = params['beta_final_at']
        self.priorityTree = SumTree(self.alpha, self.capacity)
        self.curStep = 0.0
        self.beta = 0.4
        self.beta_increment_per_sample = 1.0 / (
            self.params['max_steps'] * self.params['train_episode_count']
        )
        self.impSamplingWeights = []
        self.sampledMemIndexes = []

    def add_batch(self, experience):
        ReplayBuffer.add_batch(self, experience)
        self.priorityTree.addNew(self.priorityTree.getMaxPriority())

    def getISW(self):
        return self.impSamplingWeights

    def update(self,deltas):
        for i,memIdx in enumerate(self.sampledMemIndexes):
            new_priority  = math.fabs(deltas[i][0].numpy()) + self.epsilon
            self.priorityTree.updateTree(memIdx, new_priority)

    def get_next(self, batch_size):
        if self.num_experiences < batch_size:
            batch_size = self.num_experiences
        pTotal = self.priorityTree.getSigmaPriority()
        pTot_by_k = int(pTotal // batch_size)
        self.sampledMemIndexes = []
        self.impSamplingWeights = []
        self.beta = min(1.0, self.beta + self.beta_increment_per_sample)
        for j in range(batch_size):
            lower_bound = j * (pTot_by_k)
            upper_bound =  (j+1) * (pTot_by_k)
            sampledVal = random.sample(range(lower_bound,upper_bound),1)

            sampledMemIdx, sampledPriority = \
                self.priorityTree.getSelectedLeaf(sampledVal[0])

            self.sampledMemIndexes.append(sampledMemIdx)
            assert sampledPriority !=0.0, \
                "Can't progress with a sampled priority = ZERO!"

            sampledProb  = (
                sampledPriority ** self.alpha
            ) / self.priorityTree.getSigmaPriority(withAlpha= True)

            impSampleWt = (
                self.capacity * sampledProb
            ) ** (-1 * self.beta)
            self.impSamplingWeights.append(impSampleWt)

        maxISW = max(self.impSamplingWeights)
        self.impSamplingWeights[:] = [
            x / maxISW for x in self.impSamplingWeights
        ]

        out = []
        for idx in self.sampledMemIndexes:
            out.append(self.buffer[idx])

        return out

    def clear(self):
        ReplayBuffer.clear(self)
        self.curStep = 0.0
        self.beta = 0.0
        self.impSamplingWeights = []
        self.sampledMemIndexes = []


