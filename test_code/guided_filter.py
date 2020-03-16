import tensorflow as tf
import numpy as np




def tf_box_filter(x, r):
    k_size = int(2*r+1)
    ch = x.get_shape().as_list()[-1]
    weight = 1/(k_size**2)
    box_kernel = weight*np.ones((k_size, k_size, ch, 1))
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



def fast_guided_filter(lr_x, lr_y, hr_x, r=1, eps=1e-8):
    
    #assert lr_x.shape.ndims == 4 and lr_y.shape.ndims == 4 and hr_x.shape.ndims == 4
   
    lr_x_shape = tf.shape(lr_x)
    #lr_y_shape = tf.shape(lr_y)
    hr_x_shape = tf.shape(hr_x)
    
    N = tf_box_filter(tf.ones((1, lr_x_shape[1], lr_x_shape[2], 1), dtype=lr_x.dtype), r)

    mean_x = tf_box_filter(lr_x, r) / N
    mean_y = tf_box_filter(lr_y, r) / N
    cov_xy = tf_box_filter(lr_x * lr_y, r) / N - mean_x * mean_y
    var_x  = tf_box_filter(lr_x * lr_x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf.image.resize_images(A, hr_x_shape[1: 3])
    mean_b = tf.image.resize_images(b, hr_x_shape[1: 3])

    output = mean_A * hr_x + mean_b
    
    return output


if __name__ == '__main__':
    import cv2
    from tqdm import tqdm

    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    #input_superpixel = tf.placeholder(tf.float32, [16, 256, 256, 3])
    output = guided_filter(input_photo, input_photo, 5, eps=1)
    image = cv2.imread('output_figure1/cartoon2.jpg')
    image = image/127.5 - 1
    image = np.expand_dims(image, axis=0)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    out = sess.run(output, feed_dict={input_photo: image})
    out = (np.squeeze(out)+1)*127.5
    out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite('output_figure1/cartoon2_filter.jpg', out)
