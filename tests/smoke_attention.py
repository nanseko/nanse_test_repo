"""Smoke test optional generator attention modules."""

import os
import sys

import tensorflow as tf


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.cut_model import CUT_model


def run_attention_smoke(attention_type):
    model = CUT_model((256, 256, 3),
                      (256, 256, 3),
                      attention_type=attention_type,
                      attention_encoder=True,
                      attention_resblocks=True,
                      resnet_blocks=1)
    inputs = tf.random.normal([1, 256, 256, 3])

    outputs = model.netG(inputs, training=False)
    assert outputs.shape.as_list() == [1, 256, 256, 3]

    features = model.netE(inputs, training=False)
    assert isinstance(features, list)
    assert len(features) > 0
    assert all(len(feature.shape) == 4 for feature in features)


def main():
    run_attention_smoke('cbam')
    run_attention_smoke('coord')
    print('Attention smoke tests passed.')


if __name__ == '__main__':
    main()
