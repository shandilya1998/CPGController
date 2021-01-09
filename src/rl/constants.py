import tensorflow as tf

action_dim = 8
params = { 
    'motion_state_size'           : 10, 
    'robot_state_size'            : 10, 
    'dt'                          : 0.0001,
    'units_output_mlp'            : [10, 20, 12, action_dim],
    'units_osc'                   : 10, 
    'units_combine'               : 10, 
    'units_robot_state'           : 10, 
    'units_motion_state'          : 10,

    'BATCH_SIZE'                  : 1,
    'BUFFER_SIZE'                 : 100000,
    'GAMMA'                       : 0.99,
    'TAU'                         : 0.001,
    'LRA'                         : 0.0001,
    'LRC'                         : 0.001,
    'EXPLORE'                     : 100000,
    'train_episode_count'         : 1000,
    'test_episode_count'          : 100,
    'max_steps'                   : 40,
    'action_dim'                  : action_dim,

    'action_input_units'          : 10,
    'rnn_steps'                   : 1000,
    'critic_hidden_units'         : 10,
    'lstm_units'                  : action_dim,
    'lstm_state_dense_activation' : 'relu',

    'L0'                          : 0.01738,
    'L1'                          : 0.025677,
    'L2'                          : 0.017849,
    'L3'                          : 0.02550,
    'g'                           : -9.81,
}

observation_spec = [
    tf.TensorSpec(
        shape = (
            params['BATCH_SIZE'],
            params['rnn_steps'],
            params['motion_state_size']
        ),
        dtype = tf.dtypes.float32,
        name = 'motion state'
    ),
    tf.TensorSpec(
        shape = (
            params['BATCH_SIZE'],
            params['rnn_steps'],
            params['robot_state_size'],
        ),
        dtype = tf.dtypes.float32,
        name = 'robot state'
    ),
    tf.TensorSpec(
        shape = (
            params['BATCH_SIZE'],
            params['units_osc']
        ),
        dtype = tf.dtypes.complex64,
        name = 'oscillator state'
    ),
]

action_spec = [
    tf.TensorSpec(
           shape = (
               params['BATCH_SIZE'],
               params['rnn_steps'],
               params['action_dim']
           )
       ),
   tf.TensorSpec(
       shape = (
           params['BATCH_SIZE'],
           params['units_osc']
       ),
       dtype = tf.dtypes.complex64,
       name = 'oscillator action'
   )
]

reward_spec = [
    tf.TensorSpec(
        shape = (
            params['BATCH_SIZE'],
            params['rnn_steps']
        ),
        dtype = tf.dtypes.float32,
        name = 'reward'
    )
]

data_spec = []
data_spec.extend(observation_spec)
data_spec.extend(action_spec)
data_spec.extend(observation_spec)

data_spec.extend([
    tf.TensorSpec(
        shape = (
            params['BATCH_SIZE'],
        ),
        dtype = tf.dtypes.bool,
        name = 'done'
    )
])
