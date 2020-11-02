'''
Source code for CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'
by Xinrui Wang and Jinze yu
'''


import tensorflow as tf
import tf_slim as slim
import utils
import os
import numpy as np
import argparse
import network 
import loss
import random
from tqdm import tqdm
from guided_filter import guided_filter
random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

tf.compat.v1.disable_eager_execution()
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default = 256, type = int)
    parser.add_argument("--batch_size", default = 16, type = int)     
    parser.add_argument("--total_iter", default = 100000, type = int)
    parser.add_argument("--adv_train_lr", default = 2e-4, type = float)
    parser.add_argument("--gpu_fraction", default = 0.5, type = float)
    parser.add_argument("--save_dir", default = 'train_cartoon', type = str)
    parser.add_argument("--use_enhance", default = False)

    args = parser.parse_args()
    
    return args



def train(args):
    

    input_photo = tf.compat.v1.placeholder(tf.float32, [args.batch_size, 
                                args.patch_size, args.patch_size, 3])
    input_superpixel = tf.compat.v1.placeholder(tf.float32, [args.batch_size, 
                                args.patch_size, args.patch_size, 3])
    input_cartoon = tf.compat.v1.placeholder(tf.float32, [args.batch_size, 
                                args.patch_size, args.patch_size, 3])
    
    output = network.unet_generator(input_photo)
    output = guided_filter(input_photo, output, r=1)

    
    blur_fake = guided_filter(output, output, r=5, eps=2e-1)
    blur_cartoon = guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)

    gray_fake, gray_cartoon = utils.color_shift(output, input_cartoon)
    
    d_loss_gray, g_loss_gray = loss.lsgan_loss(network.disc_sn, gray_cartoon, gray_fake, 
                                             scale=1, patch=True, name='disc_gray')
    d_loss_blur, g_loss_blur = loss.lsgan_loss(network.disc_sn, blur_cartoon, blur_fake, 
                                             scale=1, patch=True, name='disc_blur')


    vgg_model = loss.Vgg19('train_code/vgg19_no_fc.npy')
    vgg_photo = vgg_model.build_conv4_4(input_photo)
    vgg_output = vgg_model.build_conv4_4(output)
    vgg_superpixel = vgg_model.build_conv4_4(input_superpixel)
    h, w, c = vgg_photo.get_shape().as_list()[1:]
    
    photo_loss = tf.reduce_mean(input_tensor=tf.compat.v1.losses.absolute_difference(vgg_photo, vgg_output))/(h*w*c)
    superpixel_loss = tf.reduce_mean(input_tensor=tf.compat.v1.losses.absolute_difference\
                                     (vgg_superpixel, vgg_output))/(h*w*c)
    recon_loss = photo_loss + superpixel_loss
    tv_loss = loss.total_variation_loss(output)
    
    g_loss_total = 1e4*tv_loss + 1e-1*g_loss_blur + g_loss_gray + 2e2*recon_loss
    d_loss_total = d_loss_blur + d_loss_gray

    all_vars = tf.compat.v1.trainable_variables()
    gene_vars = [var for var in all_vars if 'gene' in var.name]
    disc_vars = [var for var in all_vars if 'disc' in var.name] 
    
    
    tf.compat.v1.summary.scalar('tv_loss', tv_loss)
    tf.compat.v1.summary.scalar('photo_loss', photo_loss)
    tf.compat.v1.summary.scalar('superpixel_loss', superpixel_loss)
    tf.compat.v1.summary.scalar('recon_loss', recon_loss)
    tf.compat.v1.summary.scalar('d_loss_gray', d_loss_gray)
    tf.compat.v1.summary.scalar('g_loss_gray', g_loss_gray)
    tf.compat.v1.summary.scalar('d_loss_blur', d_loss_blur)
    tf.compat.v1.summary.scalar('g_loss_blur', g_loss_blur)
    tf.compat.v1.summary.scalar('d_loss_total', d_loss_total)
    tf.compat.v1.summary.scalar('g_loss_total', g_loss_total)
      
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        
        g_optim = tf.compat.v1.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)\
                                        .minimize(g_loss_total, var_list=gene_vars)
        
        d_optim = tf.compat.v1.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)\
                                        .minimize(d_loss_total, var_list=disc_vars)
           
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    '''
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    
    
    train_writer = tf.compat.v1.summary.FileWriter(args.save_dir+'/train_log')
    summary_op = tf.compat.v1.summary.merge_all()
    saver = tf.compat.v1.train.Saver(var_list=gene_vars, max_to_keep=20)
   
    with tf.device('/device:GPU:0'):

        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('pretrainsave_models'))

        face_photo_dir = 'dataset/face_photo'
        face_photo_list = utils.load_image_list(face_photo_dir)
        scenery_photo_dir = 'dataset/scenery_photo'
        scenery_photo_list = utils.load_image_list(scenery_photo_dir)

        face_cartoon_dir_kyoto_face = 'dataset/face_cartoon/kyoto_face/'
        face_cartoon_list = utils.load_image_list(face_cartoon_dir_kyoto_face)
        face_cartoon_dir_pa_face = 'dataset/face_cartoon/pa_face/'
        face_cartoon_list.extend(utils.load_image_list(face_cartoon_dir_pa_face))
        scenery_cartoon_dir = 'dataset/scenery_cartoon/'
        scenery_cartoon_list = utils.load_image_list(scenery_cartoon_dir+"hayao/")
        scenery_cartoon_list.extend(utils.load_image_list(scenery_cartoon_dir+"hosoda/"))
        scenery_cartoon_list = utils.load_image_list(scenery_cartoon_dir+"shinkai/")

        for total_iter in tqdm(range(args.total_iter)):

            if np.mod(total_iter, 5) == 0: 
                photo_batch = utils.next_batch(face_photo_list, args.batch_size)
                cartoon_batch = utils.next_batch(face_cartoon_list, args.batch_size)
            else:
                photo_batch = utils.next_batch(scenery_photo_list, args.batch_size)
                cartoon_batch = utils.next_batch(scenery_cartoon_list, args.batch_size)
        
            inter_out = sess.run(output, feed_dict={input_photo: photo_batch, 
                                                    input_superpixel: photo_batch,
                                                    input_cartoon: cartoon_batch})

            '''
            adaptive coloring has to be applied with the clip_by_value 
            in the last layer of generator network, which is not very stable.
            to stabiliy reproduce our results, please use power=1.0
            and comment the clip_by_value function in the network.py first
            If this works, then try to use adaptive color with clip_by_value.
            '''
            if args.use_enhance:
                superpixel_batch = utils.selective_adacolor(inter_out, power=1.2)
            else:
                superpixel_batch = utils.simple_superpixel(inter_out, seg_num=200)
                
            _, g_loss, r_loss = sess.run([g_optim, g_loss_total, recon_loss],  
                                            feed_dict={input_photo: photo_batch, 
                                                    input_superpixel: superpixel_batch,
                                                    input_cartoon: cartoon_batch})

            _, d_loss, train_info = sess.run([d_optim, d_loss_total, summary_op],  
                                            feed_dict={input_photo: photo_batch, 
                                                    input_superpixel: superpixel_batch,
                                                    input_cartoon: cartoon_batch})


            train_writer.add_summary(train_info, total_iter)
            
            if np.mod(total_iter+1, 50) == 0:

                print('Iter: {}, d_loss: {}, g_loss: {}, recon_loss: {}'.\
                        format(total_iter, d_loss, g_loss, r_loss))
                if np.mod(total_iter+1, 250 ) == 0:
                    saver.save(sess, args.save_dir+'/saved_models/model', 
                               write_meta_graph=False, global_step=total_iter)

                if np.mod(total_iter+1, 500 ) == 0:
                    
                    photo_face = utils.next_batch(face_photo_list, args.batch_size)
                    cartoon_face = utils.next_batch(face_cartoon_list, args.batch_size)
                    photo_scenery = utils.next_batch(scenery_photo_list, args.batch_size)
                    cartoon_scenery = utils.next_batch(scenery_cartoon_list, args.batch_size)

                    result_face = sess.run(output, feed_dict={input_photo: photo_face, 
                                                            input_superpixel: photo_face,
                                                            input_cartoon: cartoon_face})

                    result_scenery = sess.run(output, feed_dict={input_photo: photo_scenery, 
                                                                input_superpixel: photo_scenery,
                                                                input_cartoon: cartoon_scenery})

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
