'''
CVPR 2020 submission, Paper ID 6791
Source code for 'Learning to Cartoonize Using White-Box Cartoon Representations'
'''


import numpy as np
import scipy.stats as st
import tensorflow as tf



VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    
    def __init__(self, vgg19_npy_path=None):
        
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        print('Finished loading vgg19.npy')


    def build_conv4_4(self, rgb, include_fc=False):
        
        rgb_scaled = (rgb+1) * 127.5

        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0],
                        green - VGG_MEAN[1], red - VGG_MEAN[2]])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.relu1_1 = tf.nn.relu(self.conv1_1)
        self.conv1_2 = self.conv_layer(self.relu1_1, "conv1_2")
        self.relu1_2 = tf.nn.relu(self.conv1_2)
        self.pool1 = self.max_pool(self.relu1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.relu2_1 = tf.nn.relu(self.conv2_1)
        self.conv2_2 = self.conv_layer(self.relu2_1, "conv2_2")
        self.relu2_2 = tf.nn.relu(self.conv2_2)
        self.pool2 = self.max_pool(self.relu2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.relu3_1 = tf.nn.relu(self.conv3_1)
        self.conv3_2 = self.conv_layer(self.relu3_1, "conv3_2")
        self.relu3_2 = tf.nn.relu(self.conv3_2)
        self.conv3_3 = self.conv_layer(self.relu3_2, "conv3_3")
        self.relu3_3 = tf.nn.relu(self.conv3_3)
        self.conv3_4 = self.conv_layer(self.relu3_3, "conv3_4")
        self.relu3_4 = tf.nn.relu(self.conv3_4)
        self.pool3 = self.max_pool(self.relu3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.relu4_1 = tf.nn.relu(self.conv4_1)
        self.conv4_2 = self.conv_layer(self.relu4_1, "conv4_2")
        self.relu4_2 = tf.nn.relu(self.conv4_2)
        self.conv4_3 = self.conv_layer(self.relu4_2, "conv4_3")
        self.relu4_3 = tf.nn.relu(self.conv4_3)
        self.conv4_4 = self.conv_layer(self.relu4_3, "conv4_4")
        self.relu4_4 = tf.nn.relu(self.conv4_4)
        self.pool4 = self.max_pool(self.relu4_4, 'pool4')

        return self.conv4_4

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], 
                    strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            #relu = tf.nn.relu(bias)
            return bias



    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")



def vggloss_4_4(image_a, image_b):
    vgg_model = Vgg19('vgg19_no_fc.npy')
    vgg_a = vgg_model.build_conv4_4(image_a)
    vgg_b = vgg_model.build_conv4_4(image_b)
    VGG_loss = tf.losses.absolute_difference(vgg_a, vgg_b)
    #VGG_loss = tf.nn.l2_loss(vgg_a - vgg_b)
    h, w, c= vgg_a.get_shape().as_list()[1:]
    VGG_loss = tf.reduce_mean(VGG_loss)/(h*w*c)
    return VGG_loss



def wgan_loss(discriminator, real, fake, patch=True, 
              channel=32, name='discriminator', lambda_=2):
    real_logits = discriminator(real, patch=patch, channel=channel, name=name, reuse=False)
    fake_logits = discriminator(fake, patch=patch, channel=channel, name=name, reuse=True)

    d_loss_real = - tf.reduce_mean(real_logits)
    d_loss_fake = tf.reduce_mean(fake_logits)

    d_loss = d_loss_real + d_loss_fake
    g_loss = - d_loss_fake

    """ Gradient Penalty """
    # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
    alpha = tf.random_uniform([tf.shape(real)[0], 1, 1, 1], minval=0.,maxval=1.)
    differences = fake - real # This is different from MAGAN
    interpolates = real + (alpha * differences)
    inter_logit = discriminator(interpolates, channel=channel, name=name, reuse=True)
    gradients = tf.gradients(inter_logit, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    d_loss += lambda_ * gradient_penalty
    
    return d_loss, g_loss


def gan_loss(discriminator, real, fake, scale=1,channel=32, patch=False, name='discriminator'):

    real_logit = discriminator(real, scale, channel, name=name, patch=patch, reuse=False)
    fake_logit = discriminator(fake, scale, channel, name=name, patch=patch, reuse=True)

    real_logit = tf.nn.sigmoid(real_logit)
    fake_logit = tf.nn.sigmoid(fake_logit)
    
    g_loss_blur = -tf.reduce_mean(tf.log(fake_logit)) 
    d_loss_blur = -tf.reduce_mean(tf.log(real_logit) + tf.log(1. - fake_logit))

    return d_loss_blur, g_loss_blur



def lsgan_loss(discriminator, real, fake, scale=1, 
               channel=32, patch=False, name='discriminator'):
    
    real_logit = discriminator(real, scale, channel, name=name, patch=patch, reuse=False)
    fake_logit = discriminator(fake, scale, channel, name=name, patch=patch, reuse=True)

    g_loss = tf.reduce_mean((fake_logit - 1)**2)
    d_loss = 0.5*(tf.reduce_mean((real_logit - 1)**2) + tf.reduce_mean(fake_logit**2))

    return d_loss, g_loss



def total_variation_loss(image, k_size=1):
    h, w = image.get_shape().as_list()[1:3]
    tv_h = tf.reduce_mean((image[:, k_size:, :, :] - image[:, :h - k_size, :, :])**2)
    tv_w = tf.reduce_mean((image[:, :, k_size:, :] - image[:, :, :w - k_size, :])**2)
    tv_loss = (tv_h + tv_w)/(3*h*w)
    return tv_loss




if __name__ == '__main__':
    pass


    