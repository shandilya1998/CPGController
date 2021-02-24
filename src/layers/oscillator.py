import tensorflow as tf

class HopfOscillator(tf.keras.Model):
    def __init__(
        self,
        units,
        dt, 
        name = 'hopf_oscillator',
        dtype = 'complex64',
        **kwargs
    ):
        super(
            HopfOscillator,
            self
        ).__init__(
            name = name,
            dtype = dtype,
            **kwargs
        )
 
        self.units = int(units) if not isinstance(units, int) else units
        self.dt = tf.constant(float(dt) if not isinstance(dt, float) else dt)*tf.ones((self.units,), dtype = tf.dtypes.float32)
        self.range = tf.range(start = 1, limit = self.units+1, delta = 1, dtype = 'float32')

    def build(self, input_shape):

        self.state_input_shape = input_shape[0]
        self.omega_input_shape = input_shape[1]
        self.mu_input_shape = input_shape[2]
        self.bias_input_shape = input_shape[3]

        last_dim_state = tf.compat.dimension_value(
            self.state_input_shape[-1]
        )
        last_dim_omega = tf.compat.dimension_value(
            self.omega_input_shape[-1]
        )
        last_dim_mu = tf.compat.dimension_value(self.mu_input_shape[-1])
        last_dim_bias = tf.compat.dimension_value(self.bias_input_shape[-1])

        if last_dim_state is None or last_dim_omega is None or last_dim_mu is None or last_dim_bias is None:
            raise ValueError('The last dimension of the inputs to `HopfOscillator` '
                'should be defined. Found `None`.')
        if last_dim_state != self.units:
            raise ValueError('The last dimension of the state inputs to `HopfOscillator` '
                'should be equal to number of units. Found `{dim}`.'.format(dim = last_dim_state))

        if last_dim_omega != 1:
            raise ValueError('The last dimension of the omega inputs to `HopfOscillator` '
                'should be equal to 1. Found `{dim}`.'.format(dim = last_dim_omega))

        if last_dim_mu != self.units:
            raise ValueError('The last dimension of the mu inputs to `HopfOscillator` '
                'should be equal to number of units. Found `{dim}`.'.format(dim = last_dim_mu))

        if last_dim_bias != self.units:
            raise ValueError('The last dimension of the bias inputs to `HopfOscillator` '
                'should be equal to number of units. Found `{dim}`.'.format(dim = last_dim_bias))

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs):
        """
            inputs : [
                state : (None, 2 * units),
                omega : (None, 1),
                mu : (None, units),
                b : (None, units)
            ]
        """
        input_dim inputs[0].shape // 2
        real_state = inputs[0][:, :input_dim]
        img_state = inputs[0][:, input_dim:]
        r = tf.math.sqrt(tf.math.square(reat_state), tf.math.square(imag_state))
        phi = tf.math.atan2(imag_state, real_state)
        r = r + (inputs[2] - r*r)*r*self.dt + inputs[3]
        phi = phi + inputs[1]*self.range*self.dt
        Z_real = tf.math.multiply*(r, tf.math.cos(phi))
        Z_imag = tf.math.mmultiply(r, tf.math.sin(phi))
        return tf.concat([Z_real, Z_imag], -1)

