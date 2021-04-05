import tensorflow as tf
import numpy as np

class ComplexInitializer(tf.keras.initializers.Initializer):
    def __init__(self, initializer, name = 'complex_initializer'):
        self.initializer = initializer
        self.name = name

    def __call__(self, shape, dtype=tf.dtypes.complex64):
        return tf.complex(self.initializer(shape), self.initializer(shape))

    def get_config(self):  # To support serialization
        return {
            "initializer": tf.keras.initializer.serialize(self.initializer),
            "name": self.name
        }

def relu(x):
    return tf.complex(tf.nn.relu(tf.math.real(x)), tf.nn.relu(tf.math.imag(x)))

def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape.
        Args:
            shape: Integer shape tuple or TF tensor shape.
        Returns:
            A tuple of integer scalars (fan_in, fan_out).
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)

class ComplexDense(tf.keras.layers.Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 init_criterion='he',
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ComplexDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.init_criterion = init_criterion
        if kernel_initializer in {'complex'}:
            self.kernel_initializer = kernel_initializer
        else:
            self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed
        self.input_spec = tf.keras.layers.InputSpec(ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[-1] % 2 == 0
        input_dim = input_shape[ -1] // 2
        data_format = tf.keras.backend.image_data_format()
        kernel_shape = (input_dim, self.units)
        fan_in, fan_out = _compute_fans(
            kernel_shape,
        )
        if self.init_criterion == 'he':
            s = tf.math.sqrt(1. / fan_in)
        elif self.init_criterion == 'glorot':
            s = tf.math.sqrt(1. / (fan_in + fan_out))

        # Initialization using euclidean representation:
        def init_w_real(shape, dtype=None):
            return tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=s, seed=self.seed
            )(shape)

        def init_w_imag(shape, dtype=None):
            return tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=s, seed=self.seed
            )(shape)

        if self.kernel_initializer in {'complex'}:
            real_init = init_w_real
            imag_init = init_w_imag
        else:
            real_init = self.kernel_initializer
            imag_init = self.kernel_initializer

        self.real_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=real_init,
            name='real_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self.imag_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=imag_init,
            name='imag_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(2 * self.units,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        self.input_spec = tf.keras.layers.InputSpec(ndim=2, axes={-1: 2 * input_dim})
        self.built = True

    def call(self, inputs):
        input_shape = inputs.shape
        input_dim = input_shape[-1] // 2
        real_input = inputs[:, :input_dim]
        imag_input = inputs[:, input_dim:]

        cat_kernels_4_real = tf.concat(
            [self.real_kernel, -self.imag_kernel],
            axis=-1
        )
        cat_kernels_4_imag = tf.concat(
            [self.imag_kernel, self.real_kernel],
            axis=-1
        )
        cat_kernels_4_complex = tf.concat(
            [cat_kernels_4_real, cat_kernels_4_imag],
            axis=0
        )

        output = tf.keras.backend.dot(inputs, cat_kernels_4_complex)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 2 * self.units
        return tuple(output_shape)

    def get_config(self):
        if self.kernel_initializer in {'complex'}:
            ki = self.kernel_initializer
        else:
            ki = initializers.serialize(self.kernel_initializer)
        config = {
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'init_criterion': self.init_criterion,
            'kernel_initializer': ki,
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'seed': self.seed,
        }
        base_config = super(ComplexDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ComplexGRUCell(tf.keras.layers.Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 init_criterion='he',
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ComplexGRUCell, self).__init__(**kwargs)
        self.units = units
        if activation is not None:
            self.activation = tf.keras.activations.get(activation)
        else:
            self.activation = tf.keras.activations.get('tanh')
        self.sigmoid = tf.keras.activations.get('sigmoid')
        self.use_bias = use_bias
        self.init_criterion = init_criterion
        if kernel_initializer in {'complex'}:
            self.kernel_initializer = kernel_initializer
        else:
            self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed
        self.input_spec = tf.keras.layers.InputSpec(ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[-1] % 2 == 0
        input_dim = input_shape[ -1] // 2
        data_format = tf.keras.backend.image_data_format()
        _gate_kernel_shape = (input_dim + self.units, 2 * self.units)
        _candidate_kernel_shape = (input_dim + self.units, self.units)
        fan_in, fan_out = _compute_fans(
            _gate_kernel_shape,
        )
        if self.init_criterion == 'he':
            s = tf.math.sqrt(1. / fan_in)
        elif self.init_criterion == 'glorot':
            s = tf.math.sqrt(1. / (fan_in + fan_out))

        # Initialization using euclidean representation:
        def init_w_real(shape, dtype=None):
            return tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=s, seed=self.seed
            )(shape)

        def init_w_imag(shape, dtype=None):
            return tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=s, seed=self.seed
            )(shape)

        if self.kernel_initializer in {'complex'}:
            real_init = init_w_real
            imag_init = init_w_imag
        else:
            real_init = self.kernel_initializer
            imag_init = self.kernel_initializer

        self._gate_real_kernel = self.add_weight(
            shape=_gate_kernel_shape,
            initializer=real_init,
            name='real_kernel_gru_gate',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self._gate_imag_kernel = self.add_weight(
            shape=_gate_kernel_shape,
            initializer=imag_init,
            name='imag_kernel_gru_gate',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        if self.use_bias:
            self._gate_bias = self.add_weight(
                shape=(4 * self.units,),
                initializer=self.bias_initializer,
                name='bias_gru_gate',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self._gate_bias = None

        fan_in, fan_out = _compute_fans(
            _candidate_kernel_shape,
        )
        if self.init_criterion == 'he':
            s = tf.math.sqrt(1. / fan_in)
        elif self.init_criterion == 'glorot':
            s = tf.math.sqrt(1. / (fan_in + fan_out))

        # Initialization using euclidean representation:
        def init_w_real(shape, dtype=None):
            return tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=s, seed=self.seed
            )(shape)

        def init_w_imag(shape, dtype=None):
            return tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=s, seed=self.seed
            )(shape)

        if self.kernel_initializer in {'complex'}:
            real_init = init_w_real
            imag_init = init_w_imag
        else:
            real_init = self.kernel_initializer
            imag_init = self.kernel_initializer

        self._candidate_real_kernel = self.add_weight(
            shape=_candidate_kernel_shape,
            initializer=real_init,
            name='real_kernel_gru_candidate',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self._candidate_imag_kernel = self.add_weight(
            shape=_candidate_kernel_shape,
            initializer=imag_init,
            name='imag_kernel_gru_candidate',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        if self.use_bias:
            self._candidate_bias = self.add_weight(
                shape=(2 * self.units,),
                initializer=self.bias_initializer,
                name='bias_gru_candidate',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self._gate_bias = None

        self.input_spec = tf.keras.layers.InputSpec(ndim=2, axes={-1: 2 * input_dim})
        self.built = True

    def call(self, inputs, state):
        input_shape = inputs.shape
        input_dim = input_shape[-1] // 2
        real_inputs = inputs[:, :input_dim]
        imag_inputs = inputs[:, input_dim:]

        real_state = state[:, :self.units]
        imag_state = state[:, self.units:]

        _gate_inputs = tf.concat([
            real_inputs,
            real_state,
            imag_inputs,
            imag_state
        ], -1)

        cat_gate_kernels_4_real = tf.concat(
            [self._gate_real_kernel, -self._gate_imag_kernel],
            axis=-1
        )
        cat_gate_kernels_4_imag = tf.concat(
            [self._gate_imag_kernel, self._gate_real_kernel],
            axis=-1
        )
        cat_gate_kernels_4_complex = tf.concat(
            [cat_gate_kernels_4_real, cat_gate_kernels_4_imag],
            axis=0
        )

        _gate_inputs = tf.keras.backend.dot(
            _gate_inputs,
            cat_gate_kernels_4_complex
        )
        if self.use_bias:
            _gate_inputs = tf.nn.bias_add(_gate_inputs, self._gate_bias)

        value = self.sigmoid(_gate_inputs)

        r_real, u_real, r_imag, u_imag = tf.split(
            value=value,
            num_or_size_splits=4, 
            axis=--1
        )

        r = tf.concat([r_real, r_imag], -1)
        u = tf.concat([u_real, u_imag], -1)
        r_state = r * state

        real_r_state = r_state[:, :self.units]
        imag_r_state = r_state[:, self.units:]

        _candidate_inputs = tf.concat([
            real_inputs,
            real_r_state,
            imag_inputs,
            imag_r_state
        ], -1)

        cat_candidate_kernels_4_real = tf.concat(
            [self._candidate_real_kernel, -self._candidate_imag_kernel],
            axis=-1
        )
        cat_candidate_kernels_4_imag = tf.concat(
            [self._candidate_imag_kernel, self._candidate_real_kernel],
            axis=-1
        )
        cat_candidate_kernels_4_complex = tf.concat(
            [cat_candidate_kernels_4_real, cat_candidate_kernels_4_imag],
            axis=0
        )

        _candidate_inputs = tf.keras.backend.dot(
            _candidate_inputs,
            cat_candidate_kernels_4_complex
        )
        if self.use_bias:
            _candidate_inputs = tf.nn.bias_add(
                _candidate_inputs,
                self._candidate_bias
            )

        c = self.activation(_candidate_inputs)
        new_h = u * state + (1 - u) * c
        return new_h, new_h
