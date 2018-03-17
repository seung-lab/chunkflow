#!/usr/bin/env python
__doc__ = """

Symmetric 3D U-Net.

(Optional)
Residual skip connections.

Kisuk Lee <kisuklee@mit.edu>, 2017-2018
Nicholas Turner <nturner@cs.princeton.edu>, 2017
"""

import collections
from collections import OrderedDict
from itertools import repeat
import math

import torch
from torch import nn
from torch.nn import functional as F


# Number of feature maps.
nfeatures = [24,32,48,72,104,144]

# Filter size.
sizes = [(3,3,3)] * len(nfeatures)

# In/out embedding.
embed_ks = (1,5,5)
embed_nin = nfeatures[0]
embed_nout = embed_nin


def _ntuple(n):
    """
    Copied from PyTorch source code (https://github.com/pytorch).
    """
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_triple = _ntuple(3)


def pad_size(kernel_size, mode):
    assert mode in ['valid', 'same', 'full']
    ks = _triple(kernel_size)
    if mode == 'valid':
        pad = (0,0,0)
    elif mode == 'same':
        assert all([x %  2 for x in ks])
        pad = tuple(x // 2 for x in ks)
    elif mode == 'full':
        pad = tuple(x - 1 for x in ks)
    return pad


def batchnorm(out_channels, use_bn, momentum=0.001):
    if use_bn:
        layer = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=momentum)
    else:
        layer = lambda x: x
    return layer


def residual_sum(x, skip, residual):
    return x + skip if residual else x


class Conv(nn.Module):
    """
    3D convolution w/ MSRA init.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal(self.conv.weight)
        if bias:
            nn.init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ConvT(nn.Module):
    """
    3D convolution transpose w/ MSRA init.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, bias=True):
        super(ConvT, self).__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)
        init.kaiming_normal(self.conv.weight)
        if bias:
            init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ConvMod(nn.Module):
    """
    Convolution module.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 activation=F.elu, residual=True, use_bn=True,
                 momentum=0.001):
        super(ConvMod, self).__init__()
        # Convolution params.
        ks = _triple(kernel_size)
        st = (1,1,1)
        pad = pad_size(ks, 'same')
        bias = not use_bn
        # Convolutions.
        self.conv1 = Conv(in_channels,  out_channels, ks, st, pad, bias)
        self.conv2 = Conv(out_channels, out_channels, ks, st, pad, bias)
        self.conv3 = Conv(out_channels, out_channels, ks, st, pad, bias)
        # BatchNorm.
        self.bn1 = batchnorm(out_channels, use_bn, momentum=momentum)
        self.bn2 = batchnorm(out_channels, use_bn, momentum=momentum)
        self.bn3 = batchnorm(out_channels, use_bn, momentum=momentum)
        # Activation function.
        self.activation = activation
        # Residual skip connection.
        self.residual = residual

    def forward(self, x):
        # Conv 1.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        skip = x
        # Conv 2.
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        # Conv 3.
        x = self.conv3(x)
        x = residual_sum(x, skip, self.residual)
        x = self.bn3(x)
        return self.activation(x)


class UpsampleMod(nn.Module):
    """
    Transposed Convolution module.
    """
    def __init__(self, in_channels, out_channels, up=(1,2,2), mode='bilinear',
                 activation=F.elu, use_bn=True, momentum=0.001):
        super(UpsampleMod, self).__init__()
        # Convolution params.
        ks = (1,1,1)
        st = (1,1,1)
        pad = (0,0,0)
        bias = True
        # Upsampling.
        if mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=up, mode='trilinear')
            self.conv = Conv(in_channels, out_channels, ks, st, pad, bias)
        elif mode == 'nearest':
            self.up = nn.Upsample(scale_factor=up, mode='nearest')
            self.conv = Conv(in_channels, out_channels, ks, st, pad, bias)
        elif mode == 'transpose':
            self.up = ConvT(in_channels, out_channels,
                            kernel_size=up, stride=up, bias=bias)
            self.conv = lambda x: x
        else:
            assert False, "unknown upsampling mode {}".format(mode)
        # BatchNorm and activation.
        self.bn = batchnorm(out_channels, use_bn, momentum=momentum)
        self.activation = activation

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x + skip)
        return self.activation(x)


class EmbeddingMod(nn.Module):
    """
    Embedding module.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 activation=F.elu):
        super(EmbeddingMod, self).__init__()
        pad = pad_size(kernel_size, 'same')
        self.conv = Conv(in_channels, out_channels, kernel_size,
                         stride=1, padding=pad, bias=True)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.conv(x))


class OutputMod(nn.Module):
    """
    Embedding -> output module.

    Args:
        in_channels (int)
        out_spec (dictionary): Output specification.
        kernel_size (int or 3-tuple, optional)
    """
    def __init__(self, in_channels, out_spec, kernel_size=1):
        super(OutputMod, self).__init__()

        # Sort outputs by name.
        self.spec = OrderedDict(sorted(out_spec.items(), key=lambda x: x[0]))

        padding = pad_size(kernel_size, 'same')
        for k, v in self.spec.items():
            out_channels = v[-4]
            conv = Conv(in_channels, out_channels, kernel_size,
                        stride=1, padding=padding, bias=True)
            setattr(self, k, conv)

    def forward(self, x):
        """
        Return an output list as "DataParallel" cannot handle an output
        dictionary.
        """
        return [getattr(self, k)(x) for k in self.spec]


class RSUNet(nn.Module):
    """Residual Symmetric U-Net (RSUNet).

    Args:
        in_spec (dictionary): Input specification.
        out_spec (dictionary): Output specification.
        depth (int): Depth/scale of U-Net.
        residual (bool, optional): Use residual skip connection?
        upsample (string, optional): Upsampling mode in
            ['bilinear', 'nearest', 'transpose']
        use_bn (bool, optional): Use batch normalization?
        momentum (float, optional): Momentum for batch normalization.

    Example:
        >>> in_spec  = {'input':(1,32,160,160)}
        >>> out_spec = {'affinity':(12,32,160,160)}
        >>> model = RSUNet(in_spec, out_spec, depth=4)
    """
    def __init__(self, in_spec, out_spec, depth,
                 residual=True, upsample='bilinear', use_bn=True,
                 momentum=0.001):
        super(RSUNet, self).__init__()
        self.residual = residual
        self.upsample = upsample
        self.use_bn   = use_bn
        self.momentum = momentum

        # Model assumes a single input.
        assert len(in_spec) == 1, "model takes a single input"
        self.in_spec = in_spec
        in_channels = list(in_spec.values())[0][0]

        # Model depth (# scales == depth + 1).
        assert depth < len(nfeatures)
        self.depth = depth

        # Input feature embedding without batchnorm.
        self.embed_in = EmbeddingMod(in_channels, embed_nin, embed_ks)
        in_channels = embed_nin

        # Contracting/downsampling pathway.
        for d in range(depth):
            fs, ks = nfeatures[d], sizes[d]
            self.add_conv_mod(d, in_channels, fs, ks)
            self.add_max_pool(d+1, fs)
            in_channels = fs

        # Bridge.
        fs, ks = nfeatures[depth], sizes[depth]
        self.add_conv_mod(depth, in_channels, fs, ks)
        in_channels = fs

        # Expanding/upsampling pathway.
        for d in reversed(range(depth)):
            fs, ks = nfeatures[d], sizes[d]
            self.add_upsample_mod(d, in_channels, fs)
            in_channels = fs
            self.add_dconv_mod(d, in_channels, fs, ks)

        # Output feature embedding without batchnorm.
        self.embed_out = EmbeddingMod(in_channels, embed_nout, embed_ks)
        in_channels = embed_nout

        # Output by spec.
        self.out_spec = out_spec
        self.output = OutputMod(in_channels, out_spec)

    def add_conv_mod(self, depth, in_channels, out_channels, kernel_size):
        name = 'convmod{}'.format(depth)
        module = ConvMod(in_channels, out_channels, kernel_size,
                         residual=self.residual, use_bn=self.use_bn,
                         momentum=self.momentum)
        self.add_module(name, module)

    def add_dconv_mod(self, depth, in_channels, out_channels, kernel_size):
        name = 'dconvmod{}'.format(depth)
        module = ConvMod(in_channels, out_channels, kernel_size,
                         residual=self.residual, use_bn=self.use_bn,
                         momentum=self.momentum)
        self.add_module(name, module)

    def add_max_pool(self, depth, in_channels, down=(1,2,2)):
        name = 'maxpool{}'.format(depth)
        module = nn.MaxPool3d(down)
        self.add_module(name, module)

    def add_upsample_mod(self, depth, in_channels, out_channels, up=(1,2,2)):
        name = 'upsample{}'.format(depth)
        module = UpsampleMod(in_channels, out_channels, up=up,
                             mode=self.upsample, use_bn=self.use_bn,
                             momentum=self.momentum)
        self.add_module(name, module)

    def forward(self, x):
        # Input feature embedding without batchnorm.
        x = self.embed_in(x)

        # Contracting/downsmapling pathway.
        skip = []
        for d in range(self.depth):
            convmod = getattr(self, 'convmod{}'.format(d))
            maxpool = getattr(self, 'maxpool{}'.format(d+1))
            x = convmod(x)
            skip.append(x)
            x = maxpool(x)

        # Bridge.
        bridge = getattr(self, 'convmod{}'.format(self.depth))
        x = bridge(x)

        # Expanding/upsampling pathway.
        for d in reversed(range(self.depth)):
            upsample = getattr(self, 'upsample{}'.format(d))
            dconvmod = getattr(self, 'dconvmod{}'.format(d))
            x = dconvmod(upsample(x, skip[d]))

        # Output feature embedding without batchnorm.
        x = self.embed_out(x)
        return self.output(x)
