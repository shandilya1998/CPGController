import tensorflow as tf
import numpy as np

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

        last_dim_state = tf.compat.dimension_value(
            self.state_input_shape[-1]
        )
        last_dim_omega = tf.compat.dimension_value(
            self.omega_input_shape[-1]
        )

        self._2pi = tf.constant(
            2*np.pi, dtype = tf.dtypes.float32
        )


        if last_dim_state is None or last_dim_omega is None or last_dim_bias is None:
            raise ValueError('The last dimension of the inputs to `HopfOscillator` '
                'should be defined. Found `None`.')
        if last_dim_state != 2*self.units:
            raise ValueError('The last dimension of the state inputs to `HopfOscillator` '
                'should be equal to number of units. Found `{dim}`.'.format(dim = last_dim_state))

        if last_dim_omega != 1:
            raise ValueError('The last dimension of the omega inputs to `HopfOscillator` '
                'should be equal to 1. Found `{dim}`.'.format(dim = last_dim_omega))

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs):
        """
            inputs : [
                state : (None, 2 * units),
                omega : (None, 1),
                b : (None, units)
            ]
        """
        input_dim = inputs[0].shape[-1] // 2
        real_state = inputs[0][:, :input_dim]
        imag_state = inputs[0][:, input_dim:]

        r2 = tf.math.add(
            tf.math.square(real_state),
            tf.math.square(imag_state)
        )

        real_state = real_state + (
            -inputs[1] * self.range * imag_state + (1 - r2) * real_state
        ) * self.dt

        imag_state = imag_state + (
            inputs[1] * self.range * real_state + (1 - r2) * imag_state
        ) * self.dt

        Z = tf.concat([real_state, imag_state], -1)

        return Z
