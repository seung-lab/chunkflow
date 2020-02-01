import time
import json
import math
import os
import random
import re
# from tempfile import NamedTemporaryFile  # used by BigArrayTask

# from backports import lzma               # used by HyperSquareTask
# import blosc                             # used by BigArrayTask
# import h5py                              # used by BigArrayTask

import numpy as np

from cloudvolume import CloudVolume, Storage
from cloudvolume.lib import min2, Vec, Bbox, mkdir

from . import downsample, downsample_scales


def downsample_and_upload(image,
                          bounds,
                          vol,
                          ds_shape,
                          mip=0,
                          max_mip: int=None,
                          axis='z',
                          skip_first=False,
                          sparse=False,
                          only_last_mip=True):
    """ 
  mip:int, the current mip level of image
  only_last_mip::bool, only save the last mip level or not. 
    In default as False, we'll save all the intermediate mip level 
  """
    ds_shape = min2(vol.volume_size, ds_shape[:3])

    # sometimes we downsample a base layer of 512x512
    # into underlying chunks of 64x64 which permits more scales
    underlying_mip = (mip + 1) if (mip + 1) in vol.available_mips else mip
    underlying_shape = vol.mip_underlying(underlying_mip).astype(np.float32)
    toidx = {'x': 0, 'y': 1, 'z': 2}
    preserved_idx = toidx[axis]
    underlying_shape[preserved_idx] = float('inf')

    # Need to use ds_shape here. Using image bounds means truncated
    # edges won't generate as many mip levels
    fullscales = downsample_scales.compute_plane_downsampling_scales(
        size=ds_shape,
        preserve_axis=axis,
        max_downsampled_size=int(min(*underlying_shape)),
    )
    if max_mip:
        fullscales = fullscales[:(max_mip-mip)]
    factors = downsample.scale_series_to_downsample_factors(fullscales)

    if len(factors) == 0:
        print(
            "No factors generated. Image Shape: {}, Downsample Shape: {}, Volume Shape: {}, Bounds: {}"
            .format(image.shape, ds_shape, vol.volume_size, bounds))

    downsamplefn = downsample.method(vol.layer_type, sparse=sparse)

    vol.mip = mip
    if not skip_first:
        vol[bounds.to_slices()] = image

    new_bounds = bounds.clone()

    for factor3 in factors:
        vol.mip += 1
        image = downsamplefn(image, factor3)
        new_bounds //= factor3
        new_bounds.maxpt = new_bounds.minpt + Vec(*image.shape[:3])
        if factor3 is factors[-1]:
            # this is the last mip level
            vol[new_bounds.to_slices()] = image
        else:
            # this is not the last mip level
            if not only_last_mip:
                vol[new_bounds.to_slices()] = image
