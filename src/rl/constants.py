action_dim = 8
params = { 
    'motion_state_size'   : 10, 
    'robot_state_size'    : 10, 
    'dt'                  : 0.0001,
    'units_output_mlp'    : [10, 20, 12, action_dim], # a list of units in all layers in output MLP
    'units_osc'           : 10, 
    'units_combine'       : 10, 
    'units_robot_state'   : 10, 
    'units_motion_state'  : 10,

    'BATCH_SIZE'          : 64,
    'BUFFER_SIZE'         : 100000,
    'GAMMA'               : 0.99,
    'TAU'                 : 0.001,
    'LRA'                 : 0.0001,
    'LRC'                 : 0.001,
    'EXPLORE'             : 100000,
    'train_episode_count' : 1000,
    'test_episode_count'  : 100,
    'max_steps'           : 40,
    'action_dim'          : action_dim,

    'action_input_units'  : 10,
    'rnn_steps'           : 1000,
    'critic_hidden_units' : 10,
}

