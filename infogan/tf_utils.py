import tensorflow as tf


def variables_in_current_scope():
    return tf.get_collection(tf.GraphKeys.VARIABLES, scope=tf.get_variable_scope().name)


def scope_variables(name):
    with tf.variable_scope(name):
        return variables_in_current_scope()


def leaky_rectify(x, leakiness=0.01):
    assert leakiness <= 1
    ret = tf.maximum(x, leakiness * x)
    return ret


def identity(x):
    return x


try:
    NOOP = tf.noop
except:
    # this changed for no reason in latest version. Danke!
    NOOP = tf.no_op()
