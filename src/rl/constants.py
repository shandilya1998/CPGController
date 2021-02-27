import tensorflow as tf
import numpy as np

action_dim = 12
params = {
    'motion_state_size'           : 6,
    'robot_state_size'            : 2*action_dim + 4 + 3 + 3,
    'dt'                          : 0.001,
    'units_output_mlp'            : [20, 30, 20, action_dim],
    'units_osc'                   : 15,
    'units_combine'               : action_dim,
    'units_robot_state'           : 20,
    'units_motion_state'          : 20,
    'units_history'               : 20,
    'BATCH_SIZE'                  : 100,
    'BUFFER_SIZE'                 : 100000,
    'GAMMA'                       : 0.99,
    'TEST_AFTER_N_EPISODES'       : 100,
    'TAU'                         : 0.001,
    'LRA'                         : 0.0001,
    'LRC'                         : 0.001,
    'EXPLORE'                     : 100000,
    'train_episode_count'         : 100,
    'test_episode_count'          : 10,
    'max_steps'                   : 10000,
    'action_dim'                  : action_dim,

    'units_action_input'          : 20,
    'rnn_steps'                   : 500,
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
}

observation_spec = [
    tf.TensorSpec(
        shape = (
            params['motion_state_size'],
        ),
        dtype = tf.dtypes.float32,
        name = 'motion state'
    ),
    tf.TensorSpec(
        shape = (
            params['robot_state_size'],
        ),
        dtype = tf.dtypes.float32,
        name = 'robot state'
    ),
    tf.TensorSpec(
        shape = (
            params['units_osc'] * 2,
        ),
        dtype = tf.dtypes.float32,
        name = 'oscillator state'
    ),
]

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
        params['rnn_steps'] - 1,
        params['action_dim']
    ),
    dtype = tf.dtypes.float32
)


data_spec = []
data_spec.extend(observation_spec)
data_spec.extend(action_spec)
data_spec.extend(reward_spec)
data_spec.append(history_spec)

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
    'history_spec' : history_spec
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

pretraining = {
    'Tst' : [
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        60, 80, 100, 120, 140, 75, 45, 30, 150, 81,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
        70, 90, 110, 130, 145, 85, 55, 40, 155, 91,
    ],
    'Tsw' : [
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        15, 20, 25, 30, 35, 18, 11, 7, 37, 20,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        20, 26, 33, 40, 46, 25, 15, 10, 50, 27,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41,
        12, 16, 20, 24, 28, 17, 9, 6, 30, 16,
        30, 40, 50, 60, 70, 32, 22, 15, 75, 41
    ],
    'theta_h' : [
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15
    ],
    'theta_k' : [
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,

        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45
    ]
}


num_data = len(pretraining['Tst'])
params.update(pretraining)
params.update({
    'num_data' : num_data,
    'pretrain_bs': 500
})
