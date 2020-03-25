'''
Source code for CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'
by Xinrui Wang and Jinze yu
'''



import tensorflow as tf
import tensorflow.contrib.slim as slim

import utils
import os
import numpy as np
import argparse
import network 
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"]="0"


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default = 256, type = int)
    parser.add_argument("--batch_size", default = 16, type = int)     
    parser.add_argument("--total_iter", default = 50000, type = int)
    parser.add_argument("--adv_train_lr", default = 2e-4, type = float)
    parser.add_argument("--gpu_fraction", default = 0.5, type = float)
    parser.add_argument("--save_dir", default = 'pretrain')

    args = parser.parse_args()
    
    return args



def train(args):
    

    input_photo = tf.placeholder(tf.float32, [args.batch_size, 
                                args.patch_size, args.patch_size, 3])
    
    output = network.unet_generator(input_photo)
    
    recon_loss = tf.reduce_mean(tf.losses.absolute_difference(input_photo, output))

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'gene' in var.name]
      
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        
        optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)\
                                        .minimize(recon_loss, var_list=gene_vars)
        
        
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver(var_list=gene_vars, max_to_keep=20)
   
    with tf.device('/device:GPU:0'):

        sess.run(tf.global_variables_initializer())
        face_photo_dir = 'dataset/photo_face'
        face_photo_list = utils.load_image_list(face_photo_dir)
        scenery_photo_dir = 'dataset/photo_scenery'
        scenery_photo_list = utils.load_image_list(scenery_photo_dir)


        for total_iter in tqdm(range(args.total_iter)):

            if np.mod(total_iter, 5) == 0: 
                photo_batch = utils.next_batch(face_photo_list, args.batch_size)
            else:
                photo_batch = utils.next_batch(scenery_photo_list, args.batch_size)
                
            _, r_loss = sess.run([optim, recon_loss], feed_dict={input_photo: photo_batch})

            if np.mod(total_iter+1, 50) == 0:

                print('pretrain, iter: {}, recon_loss: {}'.format(total_iter, r_loss))
                if np.mod(total_iter+1, 500 ) == 0:
                    saver.save(sess, args.save_dir+'save_models/model', 
                               write_meta_graph=False, global_step=total_iter)
                     
                    photo_face = utils.next_batch(face_photo_list, args.batch_size)
                    photo_scenery = utils.next_batch(scenery_photo_list, args.batch_size)

                    result_face = sess.run(output, feed_dict={input_photo: photo_face})
                   
                    result_scenery = sess.run(output, feed_dict={input_photo: photo_scenery})

                    utils.write_batch_image(result_face, args.save_dir+'/images', 
                                            str(total_iter)+'_face_result.jpg', 4)
                    utils.write_batch_image(photo_face, args.save_dir+'/images', 
                                            str(total_iter)+'_face_photo.jpg', 4)
                    utils.write_batch_image(result_scenery, args.save_dir+'/images', 
                                            str(total_iter)+'_scenery_result.jpg', 4)
                    utils.write_batch_image(photo_scenery, args.save_dir+'/images', 
                                            str(total_iter)+'_scenery_photo.jpg', 4)

                    

 
            
if __name__ == '__main__':
    
    args = arg_parser()
    train(args)  
   