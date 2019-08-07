#!/usr/bin/env python
import numpy as np


class PatchMask(np.ndarray):
    def __new__(cls, patch_size, overlap):
        assert len(patch_size) == 3
        assert len(overlap) == 3

        mask = make_patch_mask(patch_size, overlap)
        return np.asarray(mask).view(cls)


def make_patch_mask(patch_size, overlap):
    """
        _make_mask()
    return:
        an numpy array with data type of float32. The value was generated
        using a bump function. the overlapping borders and corners were
        normalized according to weight accumulation.
        https://en.wikipedia.org/wiki/Bump_function
    """
    bump_map = make_bump_map(patch_size)
    stride = tuple(p - o for p, o in zip(patch_size, overlap))
    # use 3x3x3 mask addition to figure out the normalization parameter
    # this is a simulation of blending
    base_mask = np.zeros(
        tuple(f + 2 * s for (f, s) in zip(patch_size, stride)),
        dtype='float64')
    for nz in range(3):
        for ny in range(3):
            for nx in range(3):
                base_mask[nz*stride[0]:nz*stride[0]+patch_size[0],
                          ny*stride[1]:ny*stride[1]+patch_size[1],
                          nx*stride[2]:nx*stride[2]+patch_size[2]] += \
                    bump_map

    bump_map /= base_mask[stride[0]:stride[0] +
                          patch_size[0], stride[1]:stride[1] +
                          patch_size[1], stride[2]:stride[2] + patch_size[2]]

    return np.asarray(bump_map, dtype='float32')


def make_bump_map(patch_size):
    x = range(patch_size[-1])
    y = range(patch_size[-2])
    z = range(patch_size[-3])
    zv, yv, xv = np.meshgrid(z, y, x, indexing='ij')
    xv = (xv + 1.0) / (patch_size[-1] + 1.0) * 2.0 - 1.0
    yv = (yv + 1.0) / (patch_size[-2] + 1.0) * 2.0 - 1.0
    zv = (zv + 1.0) / (patch_size[-3] + 1.0) * 2.0 - 1.0
    bump_map = np.exp(-1.0 / (1.0 - xv * xv) + -1.0 / (1.0 - yv * yv) +
                      -1.0 / (1.0 - zv * zv))

    # make the low value a little bit higher to avoid floating point error
    threshold = np.max(bump_map) * 1e-5
    bump_map[bump_map < threshold] = threshold
    return np.asarray(bump_map, dtype='float64')


if __name__ == '__main__':
    patch_size = (20, 256, 256)
    bump_map = make_bump_map(patch_size)

    from scipy.stats import describe
    print('describe bump map:')
    print(describe(bump_map, axis=None))

    patch_mask = PatchMask(patch_size, (4, 64, 64))
    print('shape of mask: {}'.format(patch_mask.shape))
    print('describe patch mask:')
    print(describe(patch_mask, axis=None))

    import h5py
    import os
    file_name = '/tmp/patch_mask.h5'
    if os.path.exists(file_name):
        os.remove(file_name)
    with h5py.File(file_name) as f:
        f['/main'] = patch_mask
