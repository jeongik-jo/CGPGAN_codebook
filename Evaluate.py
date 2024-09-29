from tensorflow import keras as kr
import tensorflow as tf
import tensorflow_probability as tfp
import HyperParameters as hp
from scipy.linalg import sqrtm
import numpy as np

enc = tf.keras.applications.InceptionV3(weights='imagenet', pooling='avg', include_top=False)


@tf.function
def _get_batch_results(gen: kr.Model, real_imgs):
    batch_size = real_imgs.shape[0]
    fake_imgs = tf.clip_by_value(gen([hp.cnd_dist_func(batch_size), hp.cnt_dist_func(batch_size)]), clip_value_min=-1, clip_value_max=1)

    real_ftrs = enc(tf.image.resize(real_imgs, [299, 299]))
    fake_ftrs = enc(tf.image.resize(fake_imgs, [299, 299]))

    return {'real_ftrs': real_ftrs, 'fake_ftrs': fake_ftrs}


def _pairwise_distances(U, V):
    norm_u = tf.reduce_sum(tf.square(U), 1)
    norm_v = tf.reduce_sum(tf.square(V), 1)

    norm_u = tf.reshape(norm_u, [-1, 1])
    norm_v = tf.reshape(norm_v, [1, -1])

    D = tf.maximum(norm_u - 2 * tf.matmul(U, V, False, True) + norm_v, 0.0)

    return D


def _get_fid(real_feats, fake_feats):
    real_features_mean = tf.reduce_mean(real_feats, axis=0)
    fake_features_mean = tf.reduce_mean(fake_feats, axis=0)

    mean_diff = tf.reduce_sum((real_features_mean - fake_features_mean) ** 2)
    real_cov, fake_cov = tfp.stats.covariance(real_feats), tfp.stats.covariance(fake_feats)
    cov_mean = sqrtm(tf.matmul(real_cov, fake_cov))

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    cov_diff = tf.linalg.trace(real_cov + fake_cov - 2.0 * cov_mean)

    fid = mean_diff + cov_diff

    return fid


@tf.function
def _get_pr(ref_feats, eval_feats, nhood_size=3):
    thresholds = -tf.math.top_k(-_pairwise_distances(ref_feats, ref_feats), k=nhood_size + 1, sorted=True)[0]
    thresholds = thresholds[tf.newaxis, :, -1]

    distance_pairs = _pairwise_distances(eval_feats, ref_feats)
    return tf.reduce_mean(tf.cast(tf.math.reduce_any(distance_pairs <= thresholds, axis=1), 'float32'))


def evaluate(gen: kr.Model, dataset):
    results = {}
    for real_imgs in dataset:
        batch_results = _get_batch_results(gen, real_imgs)
        for key in batch_results:
            try:
                results[key].append(batch_results[key])
            except KeyError:
                results[key] = [batch_results[key]]

    real_ftrs = tf.concat(results['real_ftrs'], axis=0)
    fake_ftrs = tf.concat(results['fake_ftrs'], axis=0)

    fid = _get_fid(real_ftrs, fake_ftrs)
    precision = _get_pr(real_ftrs, fake_ftrs)
    recall = _get_pr(fake_ftrs, real_ftrs)

    results = {'fid': fid, 'precision': precision, 'recall': recall}

    for key in results:
        print('%-20s:' % key, '%13.6f' % results[key].numpy())
    return results






