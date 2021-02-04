import tensorflow as tf
from collections import deque
import numpy as np
import random


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
            params['buffer_size']
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
