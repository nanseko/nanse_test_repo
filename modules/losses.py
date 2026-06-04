""" Implement the following loss functions that used in CUT/FastCUT model.
GANLoss
PatchNCELoss
gradient_loss (structure preservation)
color_moment_loss (color consistency)
"""

import tensorflow as tf


def _luminance(x):
    """ Map a [-1, 1] image to a [0, 1] single-channel luminance map. """
    x = (x + 1.0) * 0.5
    if x.shape[-1] == 3:
        return tf.image.rgb_to_grayscale(x)
    return tf.reduce_mean(x, axis=-1, keepdims=True)


def gradient_loss(source, generated, blur=True):
    """ Structure-preservation loss between input and output edges.

    Encourages the generated image to keep the spatial gradients (edges:
    roads, boundaries, coastlines) of the source. The source is optionally
    blurred first so SAR speckle does not inject spurious edges.
    Both inputs are expected in the [-1, 1] range.
    """
    gs = _luminance(source)
    gg = _luminance(generated)
    if blur:
        gs = tf.nn.avg_pool2d(gs, ksize=3, strides=1, padding='SAME')

    s_dx = gs[:, :, 1:, :] - gs[:, :, :-1, :]
    s_dy = gs[:, 1:, :, :] - gs[:, :-1, :, :]
    g_dx = gg[:, :, 1:, :] - gg[:, :, :-1, :]
    g_dy = gg[:, 1:, :, :] - gg[:, :-1, :, :]

    return tf.reduce_mean(tf.abs(s_dx - g_dx)) + tf.reduce_mean(tf.abs(s_dy - g_dy))


def color_moment_loss(generated, reference):
    """ Color-consistency loss matching per-channel mean/std (1st/2nd moments).

    Intended for the identity path (idt_B = G(real_B) vs real_B): an identity
    mapping should preserve the target-domain colour statistics. Inputs in
    the [-1, 1] range; moments are computed per image over spatial dims.
    """
    g_mean, g_var = tf.nn.moments(generated, axes=[1, 2])
    r_mean, r_var = tf.nn.moments(reference, axes=[1, 2])
    g_std = tf.sqrt(g_var + 1e-5)
    r_std = tf.sqrt(r_var + 1e-5)

    return tf.reduce_mean(tf.abs(g_mean - r_mean)) + tf.reduce_mean(tf.abs(g_std - r_std))


class GANLoss:
    def __init__(self, gan_mode):
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = tf.keras.losses.MeanSquaredError()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented.')

    def __call__(self, prediction, target_is_real):

        if self.gan_mode == 'lsgan':
            if target_is_real:
                loss = self.loss(tf.ones_like(prediction), prediction)
            else:
                loss = self.loss(tf.zeros_like(prediction), prediction)

        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = tf.reduce_mean(tf.math.softplus(-prediction))
            else:
                loss = tf.reduce_mean(tf.math.softplus(prediction))
                
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = tf.reduce_mean(-prediction)
            else:
                loss = tf.reduce_mean(prediction)
        return loss


class PatchNCELoss:
    def __init__(self, nce_temp, nce_lambda):
        # Potential: only supports for batch_size=1 now.
        self.nce_temp = nce_temp
        self.nce_lambda = nce_lambda
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(
                                        reduction=tf.keras.losses.Reduction.NONE,
                                        from_logits=True)

    def __call__(self, source, target, netE, netF):
        feat_source = netE(source, training=True)
        feat_target = netE(target, training=True)

        feat_source_pool, sample_ids = netF(feat_source, patch_ids=None, training=True)
        feat_target_pool, _ = netF(feat_target, patch_ids=sample_ids, training=True)

        total_nce_loss = 0.0
        for feat_s, feat_t in zip(feat_source_pool, feat_target_pool):
            n_patches, dim = feat_s.shape

            logit = tf.matmul(feat_s, tf.transpose(feat_t)) / self.nce_temp

            # Diagonal entries are pos logits, the others are neg logits.
            diagonal = tf.eye(n_patches, dtype=tf.bool)
            target = tf.where(diagonal, 1.0, 0.0)

            loss = self.cross_entropy_loss(target, logit) * self.nce_lambda
            total_nce_loss += tf.reduce_mean(loss)

        return total_nce_loss / len(feat_source_pool)