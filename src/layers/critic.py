import tensorflow as tf

class Critic(tf.keras.Model):
    def __init__(
        self, 
        units_combine,
        units_robot_state,
        units_motion_state,
        units_action_input,
        units_hidden1,
        activation_combine = 'relu',
        activation_robot_state = 'relu',
        activation_motion_state = 'relu',
        activation_action_input = 'relu',
        activation_h1 = 'relu',
    ):
        super(Critic, self).__init__()

        self.combine_dense = tf.keras.layers.Dense(
            units = units_combine,
            activation = activation_combine,
            dtype = 'float32',
            name = 'combine_dense'
        )
        self.robot_state_dense = tf.keras.layers.Dense(
            units = units_robot_state,
            activation = activation_robot_state,
            dtype = 'float32',
            name = 'robot_state_dense'
        )
        self.action_input_dense = tf.keras.layers.Dense(
            units = units_action_input,
            activation = activation_action_input,
            dtype = 'float32',
            name = 'action_input_dense'
        )
        self.motion_state_dense = tf.keras.layers.Dense(
            units = units_motion_state,
            activation = activation_motion_state,
            dtype = 'float32',
            name = 'motion_state_dense'
        )
        self.h1_dense = tf.keras.layers.Dense(
            units = units_hidden1,
            activation = activation_h1,
            dtype = 'float32',
            name = 'h1_dense'
        )

    def call(self, inputs):
        x1, x2, x3 = inputs
        x1 = self.motion_state_dense(x1)
        x2 = self.robot_state_dense(x2)
        x3 = self.action_input_dense(x3)
        x = tf.concat([x1, x2, x3], axis = -1)
        x = self.combine_dense(x)
        x = self.h1_dense(x)
        return x


def get_critic_net(params):
    inp1 = tf.keras.Input((params['motion_state_size']), dtype = 'float32')
    inp2 = tf.keras.Input((params['robot_state_size']), dtype = 'float32')
    inp3 = tf.keras.Input((params['action_dim']), dtype = 'float32')
    out = Critic(
        units_combine = params['units_combine'],
        units_robot_state = params['units_robot_state'],
        units_motion_state = params['units_motion_state'],
        units_action_input = params['action_input_units'],
        units_hidden1 = params['critic_hidden_units'], 
    )([inp1, inp2, inp3])
    model = tf.keras.Model(inputs = [inp1, inp2, inp3], outputs = [out])
    return model

def swap_batch_timestep(input_t):
    # Swap the batch and timestep dim for the incoming tensor.
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return tf.transpose(input_t, axes)

class TimeDistributed(tf.keras.Model):
    def __init__(self, layer, params, name = 'TimeDistributedCritic'):
        super(TimeDistributed, self).__init__(name = name)
        self.layer = layer
        self.steps = params['rnn_steps']
        self.out_dim = params['critic_hidden_units']

    def call(self, inputs):
        ta_inp1 = tf.TensorArray('float32', size = 0, dynamic_size = True)
        ta_inp2 = tf.TensorArray('float32', size = 0, dynamic_size = True)
        ta_inp3 = tf.TensorArray('float32', size = 0, dynamic_size = True)
        out = tf.TensorArray(tf.float32, size = 0, dynamic_size=True)

        inp1, inp2, inp3 = [swap_batch_timestep(inp) for inp  in  inputs]

        ta_inp1.unstack(inp1)
        ta_inp2.unstack(inp2)
        ta_inp3.unstack(inp3)    

        step = tf.constant(0)

        def cond(out, step):
            return tf.math.less(
                step, 
                tf.constant(
                    self.steps, 
                    dtype = tf.int32
                )
            )

        def body(out, step):
            inputs = [
                ta_inp1.read(step),
                ta_inp2.read(step),
                ta_inp3.read(step)
            ]
    
            out = out.write(
                step, 
                self.layer(inputs)
            )

            step = tf.math.add(step, tf.constant(1))
            return out, step

        out, step = tf.while_loop(cond, body, [out, step])
        
        out = out.stack()
        out = swap_batch_timestep(out) 

        out = tf.ensure_shape(out, tf.TensorShape((None, self.steps, self.out_dim)), name='ensure_shape_critic_time_distributed_out')
        return out
