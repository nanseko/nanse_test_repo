"""Attention layers for CUT generator feature refinement."""

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense, Layer


class ChannelAttention(Layer):
    """Channel attention weights for NHWC tensors."""
    def __init__(self, reduction=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('ChannelAttention requires a known channel dimension.')

        hidden_units = max(channels // self.reduction, 1)
        self.shared_mlp = tf.keras.Sequential([
            Dense(hidden_units, activation='relu'),
            Dense(channels),
        ])

    def call(self, inputs, training=None):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2])
        max_pool = tf.reduce_max(inputs, axis=[1, 2])
        weights = tf.nn.sigmoid(self.shared_mlp(avg_pool) + self.shared_mlp(max_pool))
        weights = tf.reshape(weights, [-1, 1, 1, tf.shape(inputs)[-1]])

        return weights


class SpatialAttention(Layer):
    """Spatial attention weights for NHWC tensors."""
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = Conv2D(1, kernel_size, padding='same')

    def call(self, inputs, training=None):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        weights = tf.concat([avg_pool, max_pool], axis=-1)
        weights = tf.nn.sigmoid(self.conv(weights))

        return weights


class CBAM(Layer):
    """Convolutional Block Attention Module for NHWC tensors."""
    def __init__(self, reduction=16, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction = reduction
        self.channel_attention = ChannelAttention(reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def call(self, inputs, training=None):
        x = inputs * self.channel_attention(inputs)
        x = x * self.spatial_attention(x)

        return x


class CoordinateAttention(Layer):
    """Lightweight coordinate attention for NHWC tensors."""
    def __init__(self, reduction=16, **kwargs):
        super(CoordinateAttention, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('CoordinateAttention requires a known channel dimension.')

        bottleneck = max(channels // self.reduction, 1)
        self.bottleneck = Conv2D(bottleneck, 1, padding='same', activation='relu')
        self.conv_h = Conv2D(channels, 1, padding='same')
        self.conv_w = Conv2D(channels, 1, padding='same')

    def call(self, inputs, training=None):
        input_shape = tf.shape(inputs)
        height = input_shape[1]
        width = input_shape[2]

        pooled_h = tf.reduce_mean(inputs, axis=2, keepdims=True)
        pooled_w = tf.reduce_mean(inputs, axis=1, keepdims=True)
        pooled_w = tf.transpose(pooled_w, [0, 2, 1, 3])

        pooled = tf.concat([pooled_h, pooled_w], axis=1)
        pooled = self.bottleneck(pooled)

        attn_h = pooled[:, :height, :, :]
        attn_w = pooled[:, height:height + width, :, :]
        attn_w = tf.transpose(attn_w, [0, 2, 1, 3])

        attn_h = tf.nn.sigmoid(self.conv_h(attn_h))
        attn_w = tf.nn.sigmoid(self.conv_w(attn_w))

        return inputs * attn_h * attn_w
