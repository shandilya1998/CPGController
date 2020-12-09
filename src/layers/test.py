import tensorflow as tf
a = tf.Variable([1+2j, 3+1j, 4+5j], dtype = tf.dtypes.complex64)
b = tf.Variable([3+1j, 4-3j, 5+2j], dtype = tf.dtypes.complex64)
val = tf.Variable([3, -4, 5], dtype = tf.dtypes.complex64)
def grad():
    with tf.GradientTape() as tape:
        c = tf.complex(tf.nn.relu(tf.math.real(val*(a*a + a*(1+2j)))), tf.nn.relu(tf.math.imag(val*(a*a + a*(1+2j)))))
        d = tf.math.real(tf.math.multiply(c, b))
    grad = tape.gradient(d, [a, b, val])
    print(grad)
grad()

"""
@tf.function
def test_grad():
    c = a+b
    return tf.gradients(c, [a], grad_ys = tf.ones((3,), dtype = tf.dtypes.complex64))

print(test_grad())
"""
"""
dense = tf.keras.layers.Dense(8, tf.keras.activations.relu, dtype = tf.dtypes.complex64)
inp = tf.Variable([[1+2j,3+5j, 0+2j]], dtype = tf.dtypes.complex64)
out = dense(inp)

print(out)
"""
