import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable("u", [1, w_shape[-1]],
                        initializer=tf.random_normal_initializer(),
                        trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


def conv1d_sn(value, stride, padding, n_filters, kernels=5):
    w = tf.get_variable("kernel",
                        shape=[kernels, value.get_shape()[-1], n_filters])
    b = tf.get_variable("bias", [n_filters],
                        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv1d(value=value, filters=spectral_norm(w),
                        stride=stride, padding=padding) + b
    return conv


def res_block(name, inputs, n_filters, n_layers=2, residual_alpha=1.0,
              leaky_relu_alpha=0.2, strides=1, pad='VALID', channel='channels_last'):
    with tf.variable_scope(name):
        next_input = inputs
        for i in range(n_layers):
            with tf.variable_scope('conv' + str(i)):
                nonlinear = tf.nn.leaky_relu(next_input, alpha=leaky_relu_alpha)
                padded = tf.concat(
                    [nonlinear[:, 0:2], nonlinear, nonlinear[:, -2:]], axis=1)
                conv = conv1d_sn(padded, strides, pad, n_filters)
                next_input = conv
        return next_input * residual_alpha + inputs
