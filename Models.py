import tensorflow as tf
from tensorflow import keras as kr
import Layers
import os
import HyperParameters as hp
import numpy as np


class Gen(object):
    def build_model(self):
        cnt_vec = kr.Input([hp.cnt_dim])
        cnd_vec = kr.Input([hp.lbl_dim, hp.ctg_dim])
        return kr.Model([cnd_vec, cnt_vec], Layers.Generator()([cnd_vec, cnt_vec]))

    def __init__(self):
        self.model = self.build_model()
        self.save_cnt_vecs = hp.cnt_dist_func(hp.save_img_size)

    def save_images(self, epoch):
        if not os.path.exists('results/samples'):
            os.makedirs('results/samples')
        # --------------------------------------------------------------------------------------------------------------
        def save_fake_fixed_images():
            path = 'results/samples/fake_fixed_images'
            if not os.path.exists(path):
                os.makedirs(path)

            images = []
            for i in range(hp.cnd_sample_table.shape[0]):
                cnd_vecs = tf.one_hot(hp.cnd_sample_table[i], depth=hp.ctg_dim)
                cnd_vecs = tf.tile(cnd_vecs[tf.newaxis], [hp.save_img_size, 1, 1])
                fake_images = self.model([cnd_vecs, self.save_cnt_vecs])
                images.append(np.vstack(fake_images))

            kr.preprocessing.image.save_img(path=path + '/fake_%d.png' % epoch,
                                            x=tf.clip_by_value(np.hstack(images), clip_value_min=-1, clip_value_max=1))
        save_fake_fixed_images()
        # --------------------------------------------------------------------------------------------------------------
        def save_fake_images():
            path = 'results/samples/fake_images'
            if not os.path.exists(path):
                os.makedirs(path)
            images = []
            for i in range(16):
                fake_images = self.model([tf.tile(hp.cnd_dist_func(1), [hp.save_img_size, 1, 1]), self.save_cnt_vecs])
                images.append(np.vstack(fake_images))

            kr.preprocessing.image.save_img(path=path + '/fake_%d.png' % epoch,
                                            x=tf.clip_by_value(np.hstack(images), clip_value_min=-1, clip_value_max=1))
        save_fake_images()

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/gen.h5')
        np.save('models/ctg_probs.npy', hp.ctg_probs)

    def load(self):
        self.model.load_weights('models/gen.h5')
        hp.ctg_probs.assign(np.load('models/ctg_probs.npy'))

    def to_ema(self):
        self.train_w = [tf.constant(w) for w in self.model.trainable_variables]
        hp.gen_opt.finalize_variable_values(self.model.trainable_variables)

    def to_train(self):
        for ema_w, train_w in zip(self.model.trainable_variables, self.train_w):
            ema_w.assign(train_w)


class Dis(object):
    def build_model(self):
        inp_img = kr.Input([hp.img_res, hp.img_res, hp.img_chn])
        adv_vecs = Layers.Encoder()(inp_img)
        if not hp.is_cgpgan:
            adv_vecs = adv_vecs[:, 0, 0]
        return kr.Model(inp_img, adv_vecs)

    def __init__(self):
        self.model = self.build_model()

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/dis.h5')

    def load(self):
        self.model.load_weights('models/dis.h5')


class Cla(object):
    def build_model(self):
        input_image = kr.Input([hp.img_res, hp.img_res, hp.img_chn])
        return kr.Model(input_image, Layers.Encoder()(input_image))

    def __init__(self):
        self.model = self.build_model()

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/cla.h5')

    def load(self):
        self.model.load_weights('models/cla.h5')
