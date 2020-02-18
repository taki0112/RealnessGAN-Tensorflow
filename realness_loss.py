import tensorflow as tf
import numpy as np

# tensorflow >= 2.0

def discriminator_loss(Ra, real_logit, fake_logit):
    # Ra = Relativistic
    if Ra:
        fake_logit = tf.exp(tf.nn.log_softmax(fake_logit, axis=-1))
        real_logit = tf.exp(tf.nn.log_softmax(real_logit, axis=-1))

        num_outcomes = real_logit.shape[-1]

        gauss = np.random.normal(0, 0.1, 1000)
        count, bins = np.histogram(gauss, num_outcomes)
        anchor0 = count / sum(count)  # anchor_fake

        unif = np.random.uniform(-1, 1, 1000)
        count, bins = np.histogram(unif, num_outcomes)
        anchor1 = count / sum(count)  # anchor_real

        anchor_real = tf.zeros([real_logit.shape[0], num_outcomes]) + tf.cast(anchor1, tf.float32)
        anchor_fake = tf.zeros([real_logit.shape[0], num_outcomes]) + tf.cast(anchor0, tf.float32)

        real_loss = realness_loss(anchor_real, real_logit, skewness=10.0)
        fake_loss = realness_loss(anchor_fake, fake_logit, skewness=-10.0)

    else:
        fake_logit = tf.exp(tf.nn.log_softmax(fake_logit, axis=-1))
        real_logit = tf.exp(tf.nn.log_softmax(real_logit, axis=-1))

        num_outcomes = real_logit.shape[-1]

        gauss = np.random.normal(0, 0.1, 1000)
        count, bins = np.histogram(gauss, num_outcomes)
        anchor0 = count / sum(count)  # anchor_fake

        unif = np.random.uniform(-1, 1, 1000)
        count, bins = np.histogram(unif, num_outcomes)
        anchor1 = count / sum(count)  # anchor_real

        anchor_real = tf.zeros([real_logit.shape[0], num_outcomes]) + tf.cast(anchor1, tf.float32)
        anchor_fake = tf.zeros([real_logit.shape[0], num_outcomes]) + tf.cast(anchor0, tf.float32)

        real_loss = realness_loss(anchor_real, real_logit, skewness=10.0)
        fake_loss = realness_loss(anchor_fake, fake_logit, skewness=-10.0)

    loss = real_loss + fake_loss

    return loss


def generator_loss(Ra, real_logit, fake_logit):
    # Ra = Relativistic

    if Ra:
        fake_logit = tf.exp(tf.nn.log_softmax(fake_logit, axis=-1))
        real_logit = tf.exp(tf.nn.log_softmax(real_logit, axis=-1))

        fake_loss = realness_loss(real_logit, fake_logit)

    else:
        num_outcomes = real_logit.shape[-1]
        unif = np.random.uniform(-1, 1, 1000)
        count, bins = np.histogram(unif, num_outcomes)
        anchor1 = count / sum(count)  # anchor_real
        anchor_real = tf.zeros([real_logit.shape[0], num_outcomes]) + tf.cast(anchor1, tf.float32)

        fake_logit = tf.exp(tf.nn.log_softmax(fake_logit, axis=-1))
        fake_loss = realness_loss(anchor_real, fake_logit, skewness=10.0)

    loss = fake_loss

    return loss

def realness_loss(anchor, feature, skewness=0.0, positive_skew=10.0, negative_skew=-10.0):
    """
    num_outcomes = anchor.shape[-1]
    positive_skew = 10.0
    negative_skew = -10.0
    # [num_outcomes, positive_skew, negative_skew]
    # [51, 10.0, -10.0]
    # [21, 1.0, -1.0]

    gauss = np.random.normal(0, 0.1, 1000)
    count, bins = np.histogram(gauss, num_outcomes)
    anchor0 = count / sum(count) # anchor_fake

    unif = np.random.uniform(-1, 1, 1000)
    count, bins = np.histogram(unif, num_outcomes)
    anchor1 = count / sum(count) # anchor_real
    """

    batch_size = feature.shape[0]
    num_outcomes = feature.shape[-1]

    supports = tf.linspace(start=negative_skew, stop=positive_skew, num=num_outcomes)
    delta = (positive_skew - negative_skew) / (num_outcomes - 1)

    skew = tf.fill(dims=[batch_size, num_outcomes], value=skewness)

    # experiment to adjust KL divergence between positive/negative anchors
    Tz = skew + tf.reshape(supports, shape=[1, -1]) * tf.ones(shape=[batch_size, 1])
    Tz = tf.clip_by_value(Tz, negative_skew, positive_skew)

    b = (Tz - negative_skew) / delta
    lower_b = tf.cast(tf.math.floor(b), tf.int32).numpy()
    upper_b = tf.cast(tf.math.ceil(b), tf.int32).numpy()

    lower_b[(upper_b > 0) * (lower_b == upper_b)] -= 1
    upper_b[(lower_b < (num_outcomes - 1)) * (lower_b == upper_b)] += 1

    offset = tf.expand_dims(tf.linspace(start=0.0, stop=(batch_size - 1) * num_outcomes, num=batch_size), axis=1)
    offset = tf.tile(offset, multiples=[1, num_outcomes])

    skewed_anchor = tf.reshape(tf.zeros(shape=[batch_size, num_outcomes]), shape=[-1]).numpy()
    lower_idx = tf.cast(tf.reshape(lower_b + offset, shape=[-1]), tf.int32).numpy()
    lower_updates = tf.reshape(anchor * (tf.cast(upper_b, tf.float32) - b), shape=[-1]).numpy()
    skewed_anchor[lower_idx] += lower_updates

    upper_idx = tf.cast(tf.reshape(upper_b + offset, shape=[-1]), tf.int32).numpy()
    upper_updates = tf.reshape(anchor * (b - tf.cast(lower_b, tf.float32)), shape=[-1])
    skewed_anchor[upper_idx] += upper_updates

    loss = -(skewed_anchor * tf.reduce_mean(tf.reduce_sum(tf.math.log((feature + 1e-16)), axis=-1)))

    return loss