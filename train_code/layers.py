'''
CVPR 2020 submission, Paper ID 6791
Source code for 'Learning to Cartoonize Using White-Box Cartoon Representations'
'''


import tensorflow as tf
import numpy as np
import tf_slim as slim



def adaptive_instance_norm(content, style, epsilon=1e-5):

    c_mean, c_var = tf.nn.moments(x=content, axes=[1, 2], keepdims=True)
    s_mean, s_var = tf.nn.moments(x=style, axes=[1, 2], keepdims=True)
    c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)

    return s_std * (content - c_mean) / c_std + s_mean



def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.compat.v1.get_variable("u", [1, w_shape[-1]], 
        initializer=tf.compat.v1.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(a=w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(a=u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def conv_spectral_norm(x, channel, k_size, stride=1, name='conv_snorm'):
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable("kernel", shape=[k_size[0], k_size[1], x.get_shape()[-1], channel])
        b = tf.compat.v1.get_variable("bias", [channel], initializer=tf.compat.v1.constant_initializer(0.0))

        x = tf.nn.conv2d(input=x, filters=spectral_norm(w), strides=[1, stride, stride, 1], padding='SAME') + b

        return x



def self_attention(inputs, name='attention', reuse=False):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        h, w = tf.shape(input=inputs)[1], tf.shape(input=inputs)[2]
        bs, _, _, ch = inputs.get_shape().as_list()
        f = slim.convolution2d(inputs, ch//8, [1, 1], activation_fn=None)
        g = slim.convolution2d(inputs, ch//8, [1, 1], activation_fn=None)
        s = slim.convolution2d(inputs, 1, [1, 1], activation_fn=None)
        f_flatten = tf.reshape(f, shape=[f.shape[0], -1, f.shape[-1]])
        g_flatten = tf.reshape(g, shape=[g.shape[0], -1, g.shape[-1]])
        beta = tf.matmul(f_flatten, g_flatten, transpose_b=True)
        beta = tf.nn.softmax(beta)
        
        s_flatten = tf.reshape(s, shape=[s.shape[0], -1, s.shape[-1]])
        att_map = tf.matmul(beta, s_flatten) 
        att_map = tf.reshape(att_map, shape=[bs, h, w, 1])
        gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.compat.v1.constant_initializer(0.0))
        output = att_map * gamma + inputs
        
        return att_map, output
    

    
if __name__ == '__main__':
    pass
    
    
    
