import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp


@tf.function
def _train_step(dis: kr.Model, cla: kr.Model, gen: kr.Model, real_imgs: tf.Tensor, update_ctg_prob):
    batch_size = real_imgs.shape[0]
    cnt_vecs = hp.cnt_dist_func(batch_size)
    cnd_vecs = hp.cnd_dist_func(batch_size)
    fake_imgs = gen([cnd_vecs, cnt_vecs])

    real_ctg_probs = tf.nn.softmax(cla(real_imgs), axis=-1)
    if not update_ctg_prob:
        real_ctg_probs = real_ctg_probs - tf.reduce_mean(real_ctg_probs, axis=0, keepdims=True) + 1.0 / hp.ctg_dim
    real_ctg_vecs = tf.one_hot(tf.argmax(real_ctg_probs, axis=-1), depth=hp.ctg_dim)

    with tf.GradientTape(persistent=True) as dis_cla_tape:
        if hp.is_cgpgan:
            real_adv_vecs = tf.reduce_sum(dis(real_imgs) * real_ctg_vecs, axis=-1)
        else:
            real_adv_vecs = dis(real_imgs)

        with tf.GradientTape(persistent=True) as reg_tape:
            reg_tape.watch(fake_imgs)
            fake_ctg_logits = cla(fake_imgs)
            if hp.is_cgpgan:
                fake_adv_vecs = tf.reduce_sum(dis(fake_imgs) * cnd_vecs, axis=-1)
                adv_reg_scores = tf.reduce_mean(fake_adv_vecs, axis=-1)
            else:
                fake_adv_vecs = dis(fake_imgs)
                adv_reg_scores = fake_adv_vecs
            ctg_reg_scores = tf.reduce_mean(tf.square(1 - tf.reduce_sum(tf.nn.softmax(fake_ctg_logits, axis=-1) * cnd_vecs, axis=-1)), axis=-1)
        adv_reg_loss = tf.reduce_mean(tf.reduce_sum(tf.square(reg_tape.gradient(adv_reg_scores, fake_imgs)), axis=[1, 2, 3]))
        ctg_reg_loss = tf.reduce_mean(tf.reduce_sum(tf.square(reg_tape.gradient(ctg_reg_scores, fake_imgs)), axis=[1, 2, 3]))

        dis_adv_loss = tf.reduce_mean(tf.nn.softplus(-real_adv_vecs) + tf.nn.softplus(fake_adv_vecs))
        ctg_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(cnd_vecs, fake_ctg_logits, from_logits=True, axis=-1))

        dis_loss = dis_adv_loss + hp.adv_reg_w * adv_reg_loss
        cla_loss = hp.ctg_w * ctg_loss + hp.ctg_reg_w * ctg_reg_loss

    hp.dis_opt.minimize(dis_loss, dis.trainable_variables, tape=dis_cla_tape)
    hp.cla_opt.minimize(cla_loss, cla.trainable_variables, tape=dis_cla_tape)

    acc = 1 - tf.math.count_nonzero(tf.argmax(fake_ctg_logits, axis=-1) - tf.argmax(cnd_vecs, axis=-1)) / (batch_size * hp.lbl_dim)

    cnt_vecs = hp.cnt_dist_func(batch_size)
    cnd_vecs = hp.cnd_dist_func(batch_size)

    with tf.GradientTape() as gen_tape:
        fake_imgs = gen([cnd_vecs, cnt_vecs])
        if hp.is_cgpgan:
            fake_adv_vecs = tf.reduce_sum(dis(fake_imgs) * cnd_vecs, axis=-1)
        else:
            fake_adv_vecs = dis(fake_imgs)
        gen_loss = tf.reduce_mean(tf.nn.softplus(-fake_adv_vecs))

    hp.gen_opt.minimize(gen_loss, gen.trainable_variables, tape=gen_tape)
    hp.ctg_probs.assign(hp.ctg_probs * hp.decay_rate + tf.reduce_mean(real_ctg_probs, axis=0) * (1 - hp.decay_rate))

    results = {
        'real_adv_val': tf.reduce_mean(real_adv_vecs), 'fake_adv_val': tf.reduce_mean(fake_adv_vecs),
        'ctg_loss': ctg_loss,
        'ctg_reg_loss': ctg_reg_loss, 'adv_reg_loss': adv_reg_loss,
        'acc': acc
    }
    return results


def train(dis: kr.Model, cla: kr.Model, gen: kr.Model, dataset, epoch):
    results = {}
    for real_imgs in dataset:
        batch_results = _train_step(dis, cla, gen, real_imgs, hp.ctg_update_start_epoch <= epoch)
        for key in batch_results:
            try:
                results[key].append(batch_results[key])
            except KeyError:
                results[key] = [batch_results[key]]

    temp_results = {}
    for key in results:
        mean, variance = tf.nn.moments(tf.convert_to_tensor(results[key]), axes=0)
        temp_results[key + '_mean'] = mean
        temp_results[key + '_variance'] = variance
    temp_results['cnd_ent'] = hp.calc_cnd_ent()
    results = temp_results

    for key in results:
        print('%-30s:' % key, '%13.6f' % results[key].numpy())
    print('%-30s:' % 'ctg_prob', hp.ctg_probs.numpy())

    return results
