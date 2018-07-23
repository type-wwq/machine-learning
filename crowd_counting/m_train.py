import tensorflow as tf
import numpy as np
import os
import random
import time
from m_utils import *
from m_models import crowd_counting_cnn
from m_configs import *
from os.path import join
np.set_printoptions(threshold=np.inf)

def set_GPU(gpu='2'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

img_root_dir = './datasets/formatted_trainval/shanghaitech_part_A_patches_9/train/'
den_root_dir = './datasets/formatted_trainval/shanghaitech_part_A_patches_9/train_den/'

val_img_root_dir = './datasets/formatted_trainval/shanghaitech_part_A_patches_9/val/'
val_den_root_dir = './datasets/formatted_trainval/shanghaitech_part_A_patches_9/val_den/'

file_list = os.listdir(img_root_dir)
val_file_list = os.listdir(val_img_root_dir)
d_weight = 1.0
h_weight = 0.00004

configs = ConfigFactory()

def train():

    input_img_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    density_map_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    head_count_placeholder = tf.placeholder(tf.float32, shape=(None, 1))

    [inference_density_map, inference_head_countd] = crowd_counting_cnn(input_img_placeholder)
    density_map_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(density_map_placeholder, inference_density_map))))
    head_count_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(head_count_placeholder, inference_head_countd))))
    joint_loss = d_weight * density_map_loss + h_weight * head_count_loss
    optimizer = tf.train.MomentumOptimizer(configs.learing_rate, momentum=configs.momentum).minimize(joint_loss)

    mae = tf.reduce_mean(tf.abs(tf.subtract(head_count_placeholder, inference_head_countd)))
    mse = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(head_count_placeholder, inference_head_countd)))))

    init = tf.global_variables_initializer()
    # 启动会话
    sess = tf.InteractiveSession()
    sess.run(init)

    file_path = configs.log_router
    if (not os.path.exists(file_path)):
        os.makedirs(file_path)
    log = open(configs.log_router + configs.model_name + r'.logs', mode='a+', encoding='utf-8')

    saver = tf.train.Saver(max_to_keep=configs.max_ckpt_keep)
    ckpt = tf.train.get_checkpoint_state(configs.ckpt_router)

    if ckpt and ckpt.model_checkpoint_path:
        print('load model')
        saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(configs.iter_num):
        shuffle_filw_list = file_list
        random.shuffle(shuffle_filw_list)
        for file_index in range(len(shuffle_filw_list)):
            img_path = img_root_dir + shuffle_filw_list[file_index]
            density_map_path = den_root_dir + str(shuffle_filw_list[file_index]).split(r'.')[0] + r'.csv'
            [img, density_map, head_count] = read_datasets(img_path, density_map_path)

            feed_dict = {input_img_placeholder: img, density_map_placeholder: density_map, head_count_placeholder: head_count}

            _, h_count, j_loss, d_loss, h_loss, t_matrix_mae, t_matrix_mse = sess.run([optimizer, inference_head_countd ,joint_loss, density_map_loss, head_count_loss, mae, mse], feed_dict=feed_dict)
            format_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            format_str = 'step %d, joint loss = %.5f, density map loss = %.5f, head count loss = %5f, mae = %5f, mse = %5f, ' \
                         'inference head count =  %5f, ground truth head count = %5f'
            log_line = format_time, shuffle_filw_list[file_index], format_str % ((i + 1) * file_index, j_loss, d_loss, h_loss,
                                                                                 t_matrix_mae, t_matrix_mse, h_count, head_count)
            print(log_line)
            log.writelines(str(log_line) + '\n')

        if (not os.path.exists(configs.ckpt_router)):
            os.makedirs(configs.ckpt_router)
        saver.save(sess, configs.ckpt_router + '/v1', global_step=i)

        # for f_index in range(len(val_file_list)):
        #     val_img_path = val_img_root_dir + val_file_list[file_index]
        #     val_density_map_path = val_den_root_dir + str(val_file_list[file_index]).split(r'.')[0] + r'.csv'
        #     [val_img, val_density_map, val_head_count] = read_datasets(val_img_path, val_density_map_path)
        #     feed_dict = {input_img_placeholder: val_img, density_map_placeholder: val_density_map, head_count_placeholder: val_head_count}
        #
        #     matrix_mae, matrix_mse, j_loss, d_loss, h_loss = sess.run([mae, mse, joint_loss, density_map_loss, head_count_loss], feed_dict=feed_dict)
        #     print(matrix_mae, matrix_mse)

if __name__ == '__main__':
    train()






