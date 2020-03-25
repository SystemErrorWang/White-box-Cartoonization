'''
CVPR 2020 submission, Paper ID 6791
Source code for 'Learning to Cartoonize Using White-Box Cartoon Representations'
'''


import tensorflow as tf
import numpy as np


def tf_box_filter(x, r):
    ch = x.get_shape().as_list()[-1]
    weight = 1/((2*r+1)**2)
    box_kernel = weight*np.ones((2*r+1, 2*r+1, ch, 1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
    return output



def guided_filter(x, y, r, eps=1e-2):
    
    x_shape = tf.shape(x)
    #y_shape = tf.shape(y)

    N = tf_box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)

    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x  = tf_box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N

    output = mean_A * x + mean_b

    return output


if __name__ == '__main__':
    pass
