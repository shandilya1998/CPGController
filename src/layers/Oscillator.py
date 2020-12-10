import tensorflow as tf

class HopfOscillator(tf.keras.layers.Layer):
    def __init__(self
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
            **kwargs
        )

        self.name = name
        self.dtype = dtype
        
        self.units = int(units) if not isinstance(units, int) else units
        self.dt = tf.constant(float(dt) if not isinstance(dt, float) else dt)*tf.ones((self.units,), dtype = tf.dtypes.float32)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        self.range = tf.range(start = 1, limit = self.units+1, delta = 1)

    def build(self, input_shape):
        
        self.state_input_shape = input_shape[0]
        self.omega_input_shape = input_shape[1]
        self.mu_input_shape = input_shape[2]
        self.bias_input_shape = input_shape[3]
        
        last_dim_state = tf.compat.dimension_value(state_input_shape[-1])
        last_dim_omega = tf.compat.dimension_value(omega_input_shape[-1])
        last_dim_mu = tf.compat.dimension_value(mu_input_shape[-1])
        last_dim_bias = tf.compat.dimension_value(bias_input_shape[-1])
        
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
                state : (None, units),
                omega : (None, 1),
                mu : (None, units),
                b : (None, units)
            ]
        """
        r = tf.math.abs(inputs[0])
        phi = tf.math.angle(inputs[0])
        r = r + (inputs[2] - r*r)*r*self.dt + inputs[3]
        phi = phi + inputs[1]*self.dt
        Z = r*tf.math.exp(1j*phi)
        return Z
 
