======
Recurrent Deterministic Policy Gradient (RDPG)
======

Overview
======
`PyTorch <https://github.com/pytorch/pytorch>`_ implementation of Recurrent Deterministic Policy Gradient from the paper `Memory-based control with recurrent neural networks <https://arxiv.org/abs/1512.04455>`_ 

Requirements
  .. code-block:: console
  
    $ python -m pip install --upgrade pip
    $ pip install -r requirements.txt

Run
======
* Training:

  * Pendulum-v0

  .. code-block:: console

      $ python main.py --env Pendulum-v0 --max_episode_length 1000 --trajectory_length 10 --debug


* Testing (TODO)

References: 
======
* Memory-based control with recurrent neural networks <https://arxiv.org/abs/1512.04455>
* Continuous control with deep reinforcement learning <https://arxiv.org/abs/1509.02971>
* DDPG implementation using PyTorch <https://github.com/ghliu/pytorch-ddpg>
* PyTorch-RL <https://github.com/jingweiz/pytorch-rl>
