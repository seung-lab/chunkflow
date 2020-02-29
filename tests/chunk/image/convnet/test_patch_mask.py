#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy.stats import describe

from chunkflow.chunk.image.convnet.patch.patch_mask \
    import make_bump_map, PatchMask


def test_patch_mask():
    patch_size = (20, 256, 256)
    bump_map = make_bump_map(patch_size)

    print('describe bump map:')
    print(describe(bump_map, axis=None))

    patch_mask = PatchMask(patch_size, (4, 64, 64))
    print('shape of mask: {}'.format(patch_mask.shape))
    print('describe patch mask:')
    print(describe(patch_mask, axis=None))

    #import os
    #import h5py
    #file_name = '/tmp/patch_mask.h5'
    #if os.path.exists(file_name):
    #    os.remove(file_name)
    #with h5py.File(file_name, 'w') as f:
    #    f['/main'] = patch_mask
