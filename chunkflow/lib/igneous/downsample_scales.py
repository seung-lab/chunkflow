# @license
# Copyright 2016,2017 The Neuroglancer Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
from builtins import range

import numpy as np
import click

DEFAULT_MAX_DOWNSAMPLING = 64  # maximum factor to downsample by
DEFAULT_MAX_DOWNSAMPLED_SIZE = 128  # minimum length of a side after downsampling
DEFAULT_MAX_DOWNSAMPLING_SCALES = float('inf')


def compute_near_isotropic_downsampling_scales(
        size,
        voxel_size,
        dimensions_to_downsample,
        max_scales=DEFAULT_MAX_DOWNSAMPLING_SCALES,
        max_downsampling=DEFAULT_MAX_DOWNSAMPLING,
        max_downsampled_size=DEFAULT_MAX_DOWNSAMPLED_SIZE):
    """Compute a list of successive downsampling factors."""

    num_dims = len(voxel_size)
    cur_scale = np.ones((num_dims, ), dtype=click.INT)
    scales = [tuple(cur_scale)]
    while (len(scales) < max_scales and (np.prod(cur_scale) < max_downsampling)
           and np.all(size / cur_scale > max_downsampled_size)
           and np.all(np.mod(size, cur_scale) == 0)
           ):  # the number of chunks should be integer
        # Find dimension with smallest voxelsize.
        cur_voxel_size = cur_scale * voxel_size
        smallest_cur_voxel_size_dim = dimensions_to_downsample[np.argmin(
            cur_voxel_size[dimensions_to_downsample])]
        cur_scale[smallest_cur_voxel_size_dim] *= 2
        target_voxel_size = cur_voxel_size[smallest_cur_voxel_size_dim] * 2
        for d in dimensions_to_downsample:
            if d == smallest_cur_voxel_size_dim:
                continue
            d_voxel_size = cur_voxel_size[d]
            if abs(d_voxel_size - target_voxel_size) > abs(d_voxel_size * 2 -
                                                           target_voxel_size):
                cur_scale[d] *= 2
        scales.append(tuple(cur_scale))
    return scales


def compute_two_dimensional_near_isotropic_downsampling_scales(
        size,
        voxel_size,
        max_scales=float('inf'),
        max_downsampling=DEFAULT_MAX_DOWNSAMPLING,
        max_downsampled_size=DEFAULT_MAX_DOWNSAMPLED_SIZE):
    """Compute a list of successive downsampling factors for 2-d tiles."""

    max_scales = min(max_scales, 10)

    # First compute a set of 2-d downsamplings for XY, XZ, and YZ with a high
    # number of max_scales, and ignoring other criteria.
    scales_transpose = [
        compute_near_isotropic_downsampling_scales(
            size=size,
            voxel_size=voxel_size,
            dimensions_to_downsample=dimensions_to_downsample,
            max_scales=max_scales,
            max_downsampling=float('inf'),
            max_downsampled_size=0,
        ) for dimensions_to_downsample in [[0, 1], [0, 2], [1, 2]]
    ]

    # Truncate all list of scales to the same length, once the stopping criteria
    # is reached for all values of dimensions_to_downsample.
    scales = [((1, ) * 3, ) * 3]
    size = np.array(size)

    def scale_satisfies_criteria(scale):
        return np.prod(scale) < max_downsampling and (
            size / scale).max() > max_downsampled_size

    for i in range(1, max_scales):
        cur_scales = tuple(scales_transpose[d][i] for d in range(3))
        if all(not scale_satisfies_criteria(scale) for scale in cur_scales):
            break
        scales.append(cur_scales)
    return scales


def compute_plane_downsampling_scales(
        size,
        preserve_axis='z',
        max_scales=DEFAULT_MAX_DOWNSAMPLING_SCALES,
        max_downsampling=DEFAULT_MAX_DOWNSAMPLING,
        max_downsampled_size=DEFAULT_MAX_DOWNSAMPLED_SIZE):

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    preserve_axis = axis_map[preserve_axis]

    size = np.array(size)

    if np.any(size <= 0):
        return [(1, 1, 1)]

    size[preserve_axis] = size[(preserve_axis + 1) % 3]

    dimension = min(*size)
    num_downsamples = int(np.log2(dimension / max_downsampled_size))
    num_downsamples = min(num_downsamples, max_scales)

    factor = 2
    scales = [(1, 1, 1)]
    for i in range(num_downsamples):
        if factor > max_downsampling:
            break
        elif dimension / factor < max_downsampled_size:
            break

        scale = [factor, factor, factor]
        scale[preserve_axis] = 1

        scales.append(tuple(scale))
        factor *= 2

    return scales
