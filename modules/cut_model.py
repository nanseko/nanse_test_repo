""" Implement the following components that used in CUT/FastCUT model.
Generator (Resnet-based)
Discriminator (PatchGAN)
Encoder
PatchSampleMLP
CUT_model
"""

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda

from modules.attention import CBAM, CoordinateAttention
from modules.layers import ConvBlock, ConvTransposeBlock, ResBlock, AntialiasSampling, Padding2D
from modules.losses import GANLoss, PatchNCELoss, gradient_loss, color_moment_loss


def maybe_attention(x, attention_type, attention_reduction, name=None):
    if attention_type == 'cbam':
        return CBAM(attention_reduction, name=name)(x)
    if attention_type == 'coord':
        return CoordinateAttention(attention_reduction, name=name)(x)
    if attention_type == 'none':
        return x
    raise ValueError(f'Unsupported attention type: {attention_type}')


def default_nce_layer_names(resnet_blocks, attention_encoder):
    """ Build the default PatchNCE feature taps as stable layer *names*.

    Using names (instead of absolute layer indices) keeps the NCE taps pinned
    to the same semantic positions even when optional attention layers are
    inserted into the generator, which would otherwise shift every index.

    When ``attention_encoder`` is on, the encoder taps point at the
    attention-refined outputs so PatchNCE consumes the refined features.
    """
    enc1 = 'attention_encoder_1' if attention_encoder else 'enc_conv1'
    enc2 = 'attention_encoder_2' if attention_encoder else 'enc_conv2'
    # Mirror the original [input, conv128, conv256, resblock_0, resblock_4]
    # taps, but clamp the resblock indices to the available block count.
    res_taps = sorted({0, min(4, max(resnet_blocks - 1, 0))})
    names = ['gen_input', enc1, enc2]
    names += [f'resblock_{i}' for i in res_taps]

    return names


def Generator(input_shape,
              output_shape,
              norm_layer,
              use_antialias,
              resnet_blocks,
              impl,
              attention_type='none',
              attention_reduction=16,
              attention_encoder=False,
              attention_resblocks=False,
              attention_decoder=False):
    """ Create a Resnet-based generator.
    Adapt from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style).
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics. 
    """
    assert attention_type in ['none', 'cbam', 'coord']
    assert attention_reduction > 0
    use_bias = (norm_layer == 'instance')

    inputs = Input(shape=input_shape, name='gen_input')
    x = Padding2D(3, pad_type='reflect')(inputs)
    x = ConvBlock(64, 7, padding='valid', use_bias=use_bias, norm_layer=norm_layer, activation='relu')(x)
    if attention_encoder:
        x = maybe_attention(x, attention_type, attention_reduction, name='attention_encoder_0')

    if use_antialias:
        x = ConvBlock(128, 3, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation='relu', name='enc_conv1')(x)
        if attention_encoder:
            x = maybe_attention(x, attention_type, attention_reduction, name='attention_encoder_1')
        x = AntialiasSampling(4, mode='down', impl=impl)(x)
        x = ConvBlock(256, 3, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation='relu', name='enc_conv2')(x)
        if attention_encoder:
            x = maybe_attention(x, attention_type, attention_reduction, name='attention_encoder_2')
        x = AntialiasSampling(4, mode='down', impl=impl)(x)
    else:
        x = ConvBlock(128, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation='relu', name='enc_conv1')(x)
        if attention_encoder:
            x = maybe_attention(x, attention_type, attention_reduction, name='attention_encoder_1')
        x = ConvBlock(256, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation='relu', name='enc_conv2')(x)
        if attention_encoder:
            x = maybe_attention(x, attention_type, attention_reduction, name='attention_encoder_2')

    for block_idx in range(resnet_blocks):
        x = ResBlock(256,
                     3,
                     use_bias,
                     norm_layer,
                     attention_type=attention_type if attention_resblocks else 'none',
                     attention_reduction=attention_reduction,
                     name=f'resblock_{block_idx}')(x)

    if use_antialias:
        x = AntialiasSampling(4, mode='up', impl=impl)(x)
        x = ConvBlock(128, 3, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation='relu')(x)
        if attention_decoder:
            x = maybe_attention(x, attention_type, attention_reduction, name='attention_decoder_0')
        x = AntialiasSampling(4, mode='up', impl=impl)(x)
        x = ConvBlock(64, 3, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation='relu')(x)
        if attention_decoder:
            x = maybe_attention(x, attention_type, attention_reduction, name='attention_decoder_1')
    else:
        x = ConvTransposeBlock(128, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation='relu')(x)
        if attention_decoder:
            x = maybe_attention(x, attention_type, attention_reduction, name='attention_decoder_0')
        x = ConvTransposeBlock(64, 3, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation='relu')(x)
        if attention_decoder:
            x = maybe_attention(x, attention_type, attention_reduction, name='attention_decoder_1')

    x = Padding2D(3, pad_type='reflect')(x)
    outputs = ConvBlock(output_shape[-1], 7, padding='valid', activation='tanh')(x)

    return Model(inputs=inputs, outputs=outputs, name='generator')


def Discriminator(input_shape, norm_layer, use_antialias, impl):
    """ Create a PatchGAN discriminator.
    PatchGAN classifier described in the original pix2pix paper (https://arxiv.org/abs/1611.07004).
    Such a patch-level discriminator architecture has fewer parameters
    than a full-image discriminator and can work on arbitrarily-sized images
    in a fully convolutional fashion.
    """
    use_bias = (norm_layer == 'instance')

    inputs = Input(shape=input_shape)

    if use_antialias:
        x = ConvBlock(64, 4, padding='same', activation=tf.nn.leaky_relu)(inputs)
        x = AntialiasSampling(4, mode='down', impl=impl)(x)
        x = ConvBlock(128, 4, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation=tf.nn.leaky_relu)(x)
        x = AntialiasSampling(4, mode='down', impl=impl)(x)
        x = ConvBlock(256, 4, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation=tf.nn.leaky_relu)(x)
        x = AntialiasSampling(4, mode='down', impl=impl)(x)
    else:
        x = ConvBlock(64, 4, strides=2, padding='same', activation=tf.nn.leaky_relu)(inputs)
        x = ConvBlock(128, 4, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation=tf.nn.leaky_relu)(x)
        x = ConvBlock(256, 4, strides=2, padding='same', use_bias=use_bias, norm_layer=norm_layer, activation=tf.nn.leaky_relu)(x)

    x = Padding2D(1, pad_type='constant')(x)
    x = ConvBlock(512, 4, padding='valid', use_bias=use_bias, norm_layer=norm_layer, activation=tf.nn.leaky_relu)(x)
    x = Padding2D(1, pad_type='constant')(x)
    outputs = ConvBlock(1, 4, padding='valid')(x)

    return Model(inputs=inputs, outputs=outputs, name='discriminator')


def Encoder(generator, nce_layers):
    """ Create an Encoder that shares weights with the generator.

    ``nce_layers`` is a list of generator layer *names* (see
    ``default_nce_layer_names``). Selecting feature taps by name keeps them
    pinned to the intended positions regardless of optional attention layers.
    """
    outputs = [generator.get_layer(name).output for name in nce_layers]

    return Model(inputs=generator.input, outputs=outputs, name='encoder')


class PatchSampleMLP(Model):
    """ Create a PatchSampleMLP.
    Adapt from official CUT implementation (https://github.com/taesungp/contrastive-unpaired-translation).
    PatchSampler samples patches from pixel/feature-space.
    Two-layer MLP projects both the input and output patches to a shared embedding space.
    """
    def __init__(self, units, num_patches, **kwargs):
        super(PatchSampleMLP, self).__init__(**kwargs)
        self.units = units
        self.num_patches = num_patches
        self.l2_norm = Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + 10-10))

    def build(self, input_shape):
        initializer = tf.random_normal_initializer(0., 0.02)
        feats_shape = input_shape
        for feat_id in range(len(feats_shape)):
            mlp = tf.keras.models.Sequential([
                    Dense(self.units, activation="relu", kernel_initializer=initializer),
                    Dense(self.units, kernel_initializer=initializer),
                ])
            setattr(self, f'mlp_{feat_id}', mlp)

    def call(self, inputs, patch_ids=None, training=None):
        feats = inputs
        samples = []
        ids = []
        for feat_id, feat in enumerate(feats):
            B, H, W, C = feat.shape
     
            feat_reshape = tf.reshape(feat, [B, -1, C])

            if patch_ids is not None:
                patch_id = patch_ids[feat_id]
            else:
                patch_id = tf.random.shuffle(tf.range(H * W))[:min(self.num_patches, H * W)]

            x_sample = tf.reshape(tf.gather(feat_reshape, patch_id, axis=1), [-1, C])
            mlp = getattr(self, f'mlp_{feat_id}')
            x_sample = mlp(x_sample)
            x_sample = self.l2_norm(x_sample)
            samples.append(x_sample)
            ids.append(patch_id)

        return samples, ids


class CUT_model(Model):
    """ Create a CUT/FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020 (https://arxiv.org/abs/2007.15651).
    """
    def __init__(self,
                 source_shape,
                 target_shape,
                 cut_mode='cut',
                 gan_mode='lsgan',
                 use_antialias=True,
                 norm_layer='instance',
                 resnet_blocks=9,
                 netF_units=256,
                 netF_num_patches=256,
                 nce_temp=0.07,
                 nce_layers=None,
                 impl='ref',
                 attention_type='none',
                 attention_reduction=16,
                 attention_encoder=False,
                 attention_resblocks=False,
                 attention_decoder=False,
                 lambda_grad=0.0,
                 lambda_color=0.0,
                 **kwargs):
        assert cut_mode in ['cut', 'fastcut']
        assert gan_mode in ['lsgan', 'nonsaturating']
        assert norm_layer in [None, 'batch', 'instance']
        assert netF_units > 0
        assert netF_num_patches > 0
        assert impl in ['ref', 'cuda']
        assert attention_type in ['none', 'cbam', 'coord']
        assert attention_reduction > 0
        assert lambda_grad >= 0
        assert lambda_color >= 0
        super(CUT_model, self).__init__(self, **kwargs)

        self.lambda_grad = lambda_grad
        self.lambda_color = lambda_color
        self.gan_mode = gan_mode
        self.nce_temp = nce_temp
        if nce_layers is None:
            nce_layers = default_nce_layer_names(resnet_blocks, attention_encoder)
        self.nce_layers = nce_layers
        self.netG = Generator(source_shape,
                              target_shape,
                              norm_layer,
                              use_antialias,
                              resnet_blocks,
                              impl,
                              attention_type=attention_type,
                              attention_reduction=attention_reduction,
                              attention_encoder=attention_encoder,
                              attention_resblocks=attention_resblocks,
                              attention_decoder=attention_decoder)
        self.netD = Discriminator(target_shape, norm_layer, use_antialias, impl)
        self.netE = Encoder(self.netG, self.nce_layers)
        self.netF = PatchSampleMLP(netF_units, netF_num_patches)

        if cut_mode == 'cut':
            self.nce_lambda = 1.0
            self.use_nce_identity = True
        elif cut_mode == 'fastcut':
            self.nce_lambda = 10.0
            self.use_nce_identity = False
        else:
            raise ValueError(cut_mode)

    def compile(self,
                G_optimizer,
                F_optimizer,
                D_optimizer,):
        super(CUT_model, self).compile()
        self.G_optimizer = G_optimizer
        self.F_optimizer = F_optimizer
        self.D_optimizer = D_optimizer
        self.gan_loss_func = GANLoss(self.gan_mode)
        self.nce_loss_func = PatchNCELoss(self.nce_temp, self.nce_lambda)

    @tf.function
    def train_step(self, batch_data):
        # A is source and B is target
        real_A, real_B = batch_data
        real = tf.concat([real_A, real_B], axis=0) if self.use_nce_identity else real_A

        with tf.GradientTape(persistent=True) as tape:

            fake = self.netG(real, training=True)
            fake_B = fake[:real_A.shape[0]]
            if self.use_nce_identity:
                idt_B = fake[real_A.shape[0]:]

            """Calculate GAN loss for the discriminator"""
            fake_score = self.netD(fake_B, training=True)
            D_fake_loss = tf.reduce_mean(self.gan_loss_func(fake_score, False))

            real_score = self.netD(real_B, training=True)
            D_real_loss = tf.reduce_mean(self.gan_loss_func(real_score, True))
 
            D_loss = (D_fake_loss + D_real_loss) * 0.5

            """Calculate GAN loss and NCE loss for the generator"""
            G_loss = tf.reduce_mean(self.gan_loss_func(fake_score, True))
            NCE_loss = self.nce_loss_func(real_A, fake_B, self.netE, self.netF)

            if self.use_nce_identity:
                NCE_B_loss = self.nce_loss_func(real_B, idt_B, self.netE, self.netF)
                NCE_loss = (NCE_loss + NCE_B_loss) * 0.5

            G_loss += NCE_loss

            """Optional structure / color regularization (lambda=0 disables)."""
            if self.lambda_grad > 0:
                G_loss += self.lambda_grad * gradient_loss(real_A, fake_B)
            if self.lambda_color > 0 and self.use_nce_identity:
                G_loss += self.lambda_color * color_moment_loss(idt_B, real_B)

        D_loss_grads = tape.gradient(D_loss, self.netD.trainable_variables)
        self.D_optimizer.apply_gradients(zip(D_loss_grads, self.netD.trainable_variables))
        
        G_loss_grads = tape.gradient(G_loss, self.netG.trainable_variables)
        self.G_optimizer.apply_gradients(zip(G_loss_grads, self.netG.trainable_variables))

        F_loss_grads = tape.gradient(NCE_loss, self.netF.trainable_variables)
        self.F_optimizer.apply_gradients(zip(F_loss_grads, self.netF.trainable_variables))

        del tape
        return {'D_loss': D_loss,
                'G_loss': G_loss,
                'NCE_loss': NCE_loss}
