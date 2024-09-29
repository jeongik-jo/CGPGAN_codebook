import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow import keras as kr


dis_opt = kr.optimizers.AdamW(learning_rate=0.003, weight_decay=0.0001, beta_1=0.0, beta_2=0.99)
cla_opt = kr.optimizers.AdamW(learning_rate=0.003, weight_decay=0.0001, beta_1=0.0, beta_2=0.99)
gen_opt = kr.optimizers.AdamW(learning_rate=0.003, weight_decay=0.0001, beta_1=0.0, beta_2=0.99,
                              use_ema=True, ema_momentum=0.999, ema_overwrite_frequency=None)

is_ffhq = False
img_res = 256
img_chn = 3
cnt_dim = 1024
lbl_dim = 16
ctg_dim = 16

ctg_w = 1.0
adv_reg_w = 10.0
ctg_reg_w = 10.0
is_cgpgan = True
use_codebook = True

decay_rate = 0.999
ctg_update_start_epoch = 50

batch_size = 8
save_img_size = batch_size

train_data_size = -1
test_data_size = -1
shuffle_test_dataset = False
epochs = 150

load_model = False

eval_model = True
epoch_per_evaluate = 2

ctg_probs = tf.Variable(tf.fill([lbl_dim, ctg_dim], 1.0 / ctg_dim))

cnd_sample_table = tf.random.uniform([batch_size, lbl_dim], maxval=ctg_dim, dtype='int32')


def cnt_dist_func(batch_size):
    return tf.random.uniform([batch_size, cnt_dim], minval=-tf.sqrt(3.0), maxval=tf.sqrt(3.0))


def cnd_dist_func(batch_size):
    return tf.one_hot(tf.transpose(tf.random.categorical(logits=tf.math.log(ctg_probs + 1e-8), num_samples=batch_size)), depth=ctg_dim)


def calc_cnd_ent():
    return tf.reduce_sum(-ctg_probs * tf.math.log(ctg_probs + 1e-8))
