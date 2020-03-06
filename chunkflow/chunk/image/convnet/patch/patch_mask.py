#!/usr/bin/env python
from math import log10
import numpy as np


class PatchMask(np.ndarray):
    def __new__(cls, patch_size, overlap, dtype='float32'):
        assert len(patch_size) == 3
        assert len(overlap) == 3

        mask = make_patch_mask(patch_size, overlap, dtype=dtype)
        return np.asarray(mask).view(cls)


def make_patch_mask(patch_size, overlap, dtype='float32'):
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
    base_mask = np.zeros(tuple(f + 2 * s
                               for (f, s) in zip(patch_size, stride)),
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
    
    np.testing.assert_array_equal(bump_map[
        overlap[0]:-overlap[0], 
        overlap[1]:-overlap[1], 
        overlap[2]:-overlap[2]], 1)

    return bump_map.astype(dtype)


def make_bump_map(patch_size):
    x = range(patch_size[-1])
    y = range(patch_size[-2])
    z = range(patch_size[-3])
    zv, yv, xv = np.meshgrid(z, y, x, indexing='ij')
    xv = (xv + 1.0) / (patch_size[-1] + 1.0) * 2.0 - 1.0
    yv = (yv + 1.0) / (patch_size[-2] + 1.0) * 2.0 - 1.0
    zv = (zv + 1.0) / (patch_size[-3] + 1.0) * 2.0 - 1.0
    bump_map = np.exp(-1.0 / (1.0 - xv * xv) + 
                      -1.0 / (1.0 - yv * yv) + 
                      -1.0 / (1.0 - zv * zv))
       
    bump_map = np.interp(bump_map, (bump_map.min(), bump_map.max()), (1, 1e6))
    # make the low value a little bit higher to avoid floating point error
    #threshold = np.max(bump_map) * 1e-8
    #bump_map[bump_map < threshold] = threshold

    return np.asarray(bump_map, dtype=np.float64)
