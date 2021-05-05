import tensorflow as tf
import numpy as np

action_dim = 12
units_osc = 30
params = {
    'motion_state_size'           : 6,
    'robot_state_size'            : 4*action_dim + 4 + 8*3,
    'dt'                          : 0.001,
    'units_output_mlp'            : [60,100, action_dim],
    'units_osc'                   : units_osc,
    'units_combine_rddpg'         : [80, units_osc],
    'units_combine'               : [80, units_osc],
    'units_robot_state'           : [45, 135, units_osc],
    'units_motion_state'          : [45],
    'units_mu'                    : [45],
    'units_mean'                  : [45],
    'units_omega'                 : [45],
    'units_robot_state_critic'    : [80, 24],
    'units_gru_rddpg'             : 50,
    'units_q'                     : 1,
    'units_motion_state_critic'   : [64, 24],
    'units_action_critic'         : [64, 24],
    'units_history'               : 24,
    'BATCH_SIZE'                  : 200,
    'BUFFER_SIZE'                 : 100000,
    'WARMUP'                      : 400,
    'GAMMA'                       : 0.99,
    'TEST_AFTER_N_EPISODES'       : 25,
    'TAU'                         : 0.001,
    'decay_steps'                 : int(20),
    'LRA'                         : 5e-4,
    'LRC'                         : 5e-4,
    'EXPLORE'                     : 10000,
    'train_episode_count'         : 20000000,
    'test_episode_count'          : 10,
    'max_steps'                   : 1,
    'action_dim'                  : action_dim,

    'units_action_input'          : 20,
    'rnn_steps'                   : 12,
    'units_critic_hidden'         : 20,
    'lstm_units'                  : action_dim,
    'lstm_state_dense_activation' : 'relu',

    'L0'                          : 0.01738,
    'L1'                          : 0.025677,
    'L2'                          : 0.017849,
    'L3'                          : 0.02550,
    'g'                           : -9.81,
    'thigh'                       : 0.06200,
    'base_breadth'                : 0.04540,
    'friction_constant'           : 1e-4,
    'mu'                          : 0.001,
    'm1'                          : 0.010059,
    'm2'                          : 0.026074,
    'm3'                          : 0.007661,
    'future_steps'                : 4,
    'ou_theta'                    : 0.15,
    'ou_sigma'                    : 0.2,
    'ou_mu'                       : 0.0,
    'seed'                        : 1,
    'trajectory_length'           : 5,
    'window_length'               : 5,
    'num_validation_episodes'     : 3,
    'validate_interval'           : 1000,
}

observation_spec = [
    tf.TensorSpec(
        shape = (
            params['motion_state_size'],
        ),
        dtype = tf.dtypes.float32,
        name = 'motion_state_inp'
    ),
    tf.TensorSpec(
        shape = (
            params['robot_state_size'],
        ),
        dtype = tf.dtypes.float32,
        name = 'robot_state_inp'
    ),
    tf.TensorSpec(
        shape = (
            params['units_osc'] * 2,
        ),
        dtype = tf.dtypes.float32,
        name = 'oscillator_state_inp'
    ),
]

params_spec = [
    tf.TensorSpec(
        shape = (
            params['rnn_steps'],
            params['action_dim']
        ),
        dtype = tf.dtypes.float32,
        name = 'quadruped action'
    )
] * 2

action_spec = [
    tf.TensorSpec(
        shape = (
            params['rnn_steps'],
            params['action_dim']
        ),
        dtype = tf.dtypes.float32,
        name = 'quadruped action'
    ),
   tf.TensorSpec(
       shape = (
           params['rnn_steps'],
           params['units_osc'] * 2,
       ),
       dtype = tf.dtypes.float32,
       name = 'oscillator action'
   )
]

reward_spec = [
    tf.TensorSpec(
        shape = (),
        dtype = tf.dtypes.float32,
        name = 'reward'
    )
]

history_spec = tf.TensorSpec(
    shape = (
        2 * params['rnn_steps'] - 1,
        params['action_dim']
    ),
    dtype = tf.dtypes.float32
)

history_osc_spec = tf.TensorSpec(
    shape = (
        2 * params['rnn_steps'] - 1,
        2* params['units_osc']
    ),
    dtype = tf.dtypes.float32
)


data_spec = []
data_spec.extend(observation_spec)
data_spec.extend(action_spec)
data_spec.extend(params_spec)
data_spec.extend(reward_spec)
data_spec.append(observation_spec)

data_spec.extend([
    tf.TensorSpec(
        shape = (),
        dtype = tf.dtypes.bool,
        name = 'done'
    )
])

specs = {
    'observation_spec' : observation_spec,
    'reward_spec' : reward_spec,
    'action_spec' : action_spec,
    'data_spec' : data_spec,
    'history_spec' : history_spec,
    'history_osc_spec' : history_osc_spec
}

params.update(specs)

robot_data = {
    'leg_name_lst' : [
        'front_right_leg',
        'front_left_leg',
        'back_right_leg',
        'back_left_leg'
    ],
    'link_name_lst' :  [
        'quadruped::base_link',
        'quadruped::front_right_leg1',
        'quadruped::front_right_leg2',
        'quadruped::front_right_leg3',
        'quadruped::front_left_leg1',
        'quadruped::front_left_leg2',
        'quadruped::front_left_leg3',
        'quadruped::back_right_leg1',
        'quadruped::back_right_leg2',
        'quadruped::back_right_leg3',
        'quadruped::back_left_leg1',
        'quadruped::back_left_leg2',
        'quadruped::back_left_leg3'
    ],
    'joint_name_lst' : [
        'front_right_leg1_joint',
        'front_right_leg2_joint',
        'front_right_leg3_joint',
        'front_left_leg1_joint',
        'front_left_leg2_joint',
        'front_left_leg3_joint',
        'back_right_leg1_joint',
        'back_right_leg2_joint',
        'back_right_leg3_joint',
        'back_left_leg1_joint',
        'back_left_leg2_joint',
        'back_left_leg3_joint'
    ],
    'starting_pos' : np.array([
        -0.01, 0.01, 0.01,
        -0.01, 0.01, -0.01,
        -0.01, 0.01, -0.01,
        -0.01, 0.01, 0.01
    ], dtype = np.float32),
    'L' : 2.2*0.108,
    'W' : 2.2*0.108
}

params.update(robot_data)

Tsw = [i for i in range(80, 260, 4)]
Tsw = Tsw + Tsw + Tsw
Tst = [i*3 for i in Tsw]
Tst = Tst + Tst + Tst
theta_h = [30 for i in range(len(Tsw))] + [45 for i in range(len(Tsw))] + \
        [60 for i in range(len(Tsw))]
theta_k = [30 for i in range(len(Tsw))] + [30 for i in range(len(Tsw))] + \
        [30 for i in range(len(Tsw))]

pretraining = {
    'Tst' : Tst,
    'Tsw' : Tsw,
    'theta_h' : theta_h,
    'theta_k' : theta_k
}


num_data = 135
params.update(pretraining)
bs = 15
#num_data =135 * params['rnn_steps'] * params['max_steps']
params.update({
    'num_data' : num_data,
    'pretrain_bs': bs,
    'train_test_split' : (num_data - bs) / num_data,
    'pretrain_test_interval' : 3
})


params_ars = {
    'nb_steps'                    : 1000,
    'episode_length'              : 300,
    'learning_rate'               : 0.001,
    'nb_directions'               : 12,
    'rnn_steps'                   : 1,
    'nb_best_directions'          : 7,
    'noise'                       : 0.0003,
    'seed'                        : 1,
}

params_per = {
    'alpha'                       : 0.6,
    'epsilon'                     : 1.0,
    'beta_init'                   : 0.4,
    'beta_final'                  : 1.0,
    'beta_final_at'               : params['train_episode_count'],
    'step_size'                   : params['LRA'] / 4
}

params.update(params_per)
