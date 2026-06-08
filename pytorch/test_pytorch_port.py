""" Smoke test for the PyTorch CUT + attention port. Requires torch. """

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attention import CBAM, CoordinateAttention, make_attention
from networks import ResnetAttnGenerator
from losses_extra import gradient_loss, color_moment_loss


def test_attention_shapes():
    x = torch.randn(2, 64, 32, 32)
    assert CBAM(64)(x).shape == x.shape
    assert CoordinateAttention(64)(x).shape == x.shape
    assert make_attention('none', 64)(x).shape == x.shape
    print('attention shape: OK')


def test_generator(attention_type, **flags):
    g = ResnetAttnGenerator(3, 3, n_blocks=9, attention_type=attention_type, **flags)
    x = torch.randn(1, 3, 256, 256)
    # plain forward
    y = g(x)
    assert y.shape == (1, 3, 256, 256), y.shape
    # PatchNCE-style encode_only with the model's default taps
    nce = g.nce_default
    feats = g(x, layers=nce, encode_only=True)
    assert isinstance(feats, list) and len(feats) == len(nce)
    assert all(f.dim() == 4 for f in feats)
    print(f"generator[{attention_type}, {flags}] OK | nce_layers={nce} | "
          f"feat shapes={[tuple(f.shape[1:]) for f in feats]}")


def test_losses():
    a = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    b = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    gl = gradient_loss(a, b)
    cl = color_moment_loss(a, b)
    assert gl.item() >= 0 and cl.item() >= 0
    # gradients flow
    a = a.requires_grad_(True)
    (gradient_loss(a, b) + color_moment_loss(a, b)).backward()
    assert a.grad is not None
    print(f'losses: grad={gl.item():.4f} color={cl.item():.4f} | backward OK')


def main():
    test_attention_shapes()
    test_generator('none')
    test_generator('coord', attention_encoder=True, attention_resblocks=True)
    test_generator('cbam', attention_encoder=True, attention_resblocks=True, attention_decoder=True)
    test_generator('coord', attention_encoder=True, attention_resblocks=True, use_antialias=False)
    test_losses()
    print('\nAll PyTorch port smoke tests passed.')


if __name__ == '__main__':
    main()
