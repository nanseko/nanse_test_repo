""" PyTorch port of the attention-augmented CUT Resnet generator.

Faithful to the official CUT generator (taesungp/contrastive-unpaired-translation,
models/networks.py -> ResnetGenerator) but with optional CBAM / Coordinate
attention inserted at the encoder, inside the ResnetBlocks (residual branch),
and/or the decoder — mirroring the TensorFlow fork in modules/cut_model.py.

Key compatibility points kept identical to the official repo so PatchSampleF /
CUTModel work unchanged:
    - the generator is an nn.Sequential `self.model`
    - forward(input, layers=[], encode_only=False) taps features by layer index
    - default 9 resnet blocks, reflect padding, InstanceNorm, antialias up/down

Because attention inserts extra nn.Sequential entries, the PatchNCE tap indices
shift. The correct indices for the current configuration are exposed as
`self.nce_default` (and printed by the smoke test); set `--nce_layers` to that.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from attention import make_attention          # standalone (pytorch/)
except ImportError:                                # placed under models/ in CUT repo
    from models.attention import make_attention


# --------------------------------------------------------------------------- #
# Antialiased down / up sampling (clean reimplementation, exact 2x)
# --------------------------------------------------------------------------- #

def _binomial_kernel(filt_size=3):
    a = {
        1: [1.], 2: [1., 1.], 3: [1., 2., 1.], 4: [1., 3., 3., 1.],
        5: [1., 4., 6., 4., 1.], 6: [1., 5., 10., 10., 5., 1.],
        7: [1., 6., 15., 20., 15., 6., 1.],
    }[filt_size]
    a = np.array(a, dtype=np.float32)
    k = a[:, None] * a[None, :]
    k = k / k.sum()
    return torch.from_numpy(k)


class Downsample(nn.Module):
    """Blur (binomial) then stride-2 -> output H/2, W/2."""
    def __init__(self, channels, filt_size=3):
        super().__init__()
        self.channels = channels
        self.pad = nn.ReflectionPad2d(filt_size // 2)
        k = _binomial_kernel(filt_size)[None, None].repeat(channels, 1, 1, 1)
        self.register_buffer('filt', k)

    def forward(self, x):
        x = self.pad(x)
        return F.conv2d(x, self.filt, stride=2, groups=self.channels)


class Upsample(nn.Module):
    """Nearest 2x then binomial blur -> output H*2, W*2."""
    def __init__(self, channels, filt_size=3):
        super().__init__()
        self.channels = channels
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.pad = nn.ReflectionPad2d(filt_size // 2)
        k = _binomial_kernel(filt_size)[None, None].repeat(channels, 1, 1, 1)
        self.register_buffer('filt', k)

    def forward(self, x):
        x = self.up(x)
        x = self.pad(x)
        return F.conv2d(x, self.filt, stride=1, groups=self.channels)


# --------------------------------------------------------------------------- #
# Resnet block with optional attention
# --------------------------------------------------------------------------- #

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias, attention_type='none',
                 attention_reduction=16, attention_position='residual'):
        super().__init__()
        assert attention_position in ('residual', 'post_add')
        self.attention_position = attention_position
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            norm_layer(dim),
        )
        self.attention = make_attention(attention_type, dim, attention_reduction)

    def forward(self, x):
        residual = self.conv_block(x)
        if self.attention_position == 'residual':
            return x + self.attention(residual)
        return self.attention(x + residual)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #

class ResnetAttnGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64,
                 norm_layer=nn.InstanceNorm2d, n_blocks=9, use_antialias=True,
                 attention_type='none', attention_reduction=16,
                 attention_encoder=False, attention_resblocks=False,
                 attention_decoder=False):
        super().__init__()
        assert attention_type in ('none', 'cbam', 'coord')
        use_bias = (norm_layer == nn.InstanceNorm2d)

        model = []
        taps = {}     # semantic tap -> module index (after attention insertion)

        def add(m):
            model.append(m)
            return len(model) - 1

        def attn(ch):
            return make_attention(attention_type, ch, attention_reduction)

        # ---- stem (7x7) -> tap 'pixel' is the reflection pad (index 0) ----
        taps['pixel'] = add(nn.ReflectionPad2d(3))           # idx 0 (input pixels)
        add(nn.Conv2d(input_nc, ngf, kernel_size=7, bias=use_bias))
        add(norm_layer(ngf))
        last = add(nn.ReLU(True))
        if attention_encoder:
            last = add(attn(ngf))                            # attention_encoder_0

        # ---- downsampling x2 ----
        n_down = 2
        for i in range(n_down):
            mult = 2 ** i
            in_c, out_c = ngf * mult, ngf * mult * 2
            add(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=use_bias))
            add(norm_layer(out_c))
            last = add(nn.ReLU(True))
            if attention_encoder:
                last = add(attn(out_c))                      # attention_encoder_1/2
            taps[f'enc{i+1}'] = last                         # refined feature tap
            if use_antialias:
                add(Downsample(out_c))
            else:
                # replace previous stride-1 conv path with stride-2 (kept simple)
                add(nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1, bias=use_bias))

        # ---- resnet blocks ----
        mult = 2 ** n_down
        dim = ngf * mult
        for b in range(n_blocks):
            idx = add(ResnetBlock(
                dim, norm_layer, use_bias,
                attention_type=attention_type if attention_resblocks else 'none',
                attention_reduction=attention_reduction,
                attention_position='residual'))
            if b in (0, min(4, n_blocks - 1)):
                taps[f'res{b}'] = idx

        # ---- upsampling x2 ----
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            in_c, out_c = ngf * mult, int(ngf * mult / 2)
            if use_antialias:
                add(Upsample(in_c))
                add(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=use_bias))
            else:
                add(nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2,
                                       padding=1, output_padding=1, bias=use_bias))
            add(norm_layer(out_c))
            last = add(nn.ReLU(True))
            if attention_decoder:
                last = add(attn(out_c))                      # attention_decoder_0/1

        # ---- head ----
        add(nn.ReflectionPad2d(3))
        add(nn.Conv2d(ngf, output_nc, kernel_size=7))
        add(nn.Tanh())

        self.model = nn.Sequential(*model)
        # default PatchNCE taps (indices into self.model), attention-aware
        res_keys = ['res0', f'res{min(4, n_blocks - 1)}']
        self.nce_default = [taps['pixel'], taps['enc1'], taps['enc2']] + \
                           [taps[k] for k in res_keys]

    def forward(self, input, layers=None, encode_only=False):
        """Matches official CUT: tap features by layer index when `layers` given."""
        if layers:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return feats
            return feat, feats
        return self.model(input)
