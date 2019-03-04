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


def downsample_and_upload(
    image, bounds, vol, ds_shape, 
    mip=0, axis='z', skip_first=False,
    sparse=False, only_last_mip=False
  ): 
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
  factors = downsample.scale_series_to_downsample_factors(fullscales)

  if len(factors) == 0:
    print("No factors generated. Image Shape: {}, Downsample Shape: {}, Volume Shape: {}, Bounds: {}".format(
        image.shape, ds_shape, vol.volume_size, bounds)
    )

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


class ContrastNormalizationTask(RegisteredTask):
  """TransferTask + Contrast Correction based on LuminanceLevelsTask output."""
  # translate = change of origin

  def __init__(
    self, src_path, dest_path, levels_path, shape, 
    offset, mip, clip_fraction, fill_missing, 
    translate, minval, maxval
  ):

    super(ContrastNormalizationTask, self).__init__(
      src_path, dest_path, levels_path, shape, offset, 
      mip, clip_fraction, fill_missing, translate,
      minval, maxval
    )
    self.src_path = src_path
    self.dest_path = dest_path
    self.shape = Vec(*shape)
    self.offset = Vec(*offset)
    self.fill_missing = fill_missing
    self.translate = Vec(*translate)
    self.mip = int(mip)
    self.clip_fraction = float(clip_fraction)
    self.minval = minval 
    self.maxval = maxval

    self.levels_path = levels_path if levels_path else self.src_path

    assert 0 <= self.clip_fraction <= 1

  def execute(self):
    srccv = CloudVolume(
        self.src_path, fill_missing=self.fill_missing, mip=self.mip)
    destcv = CloudVolume(
        self.dest_path, fill_missing=self.fill_missing, mip=self.mip)

    bounds = Bbox(self.offset, self.shape[:3] + self.offset)
    bounds = Bbox.clamp(bounds, srccv.bounds)
    image = srccv[bounds.to_slices()].astype(np.float32)

    zlevels = self.fetch_z_levels()

    nbits = np.dtype(srccv.dtype).itemsize * 8
    maxval = float(2 ** nbits - 1)

    for z in range(bounds.minpt.z, bounds.maxpt.z):
      imagez = z - bounds.minpt.z
      zlevel = zlevels[imagez]
      (lower, upper) = self.find_section_clamping_values(
          zlevel, self.clip_fraction, 1 - self.clip_fraction)
      if lower == upper:
        continue
      img = image[:, :, imagez]
      img = (img - float(lower)) * (maxval / (float(upper) - float(lower)))
      image[:, :, imagez] = img

    image = np.round(image)

    minval = self.minval if self.minval is not None else 0.0
    maxval = self.maxval if self.maxval is not None else maxval

    image = np.clip(image, minval, maxval).astype(destcv.dtype)

    bounds += self.translate
    downsample_and_upload(image, bounds, destcv, self.shape, mip=self.mip)

  def find_section_clamping_values(self, zlevel, lowerfract, upperfract):
    filtered = np.copy(zlevel)

    # remove pure black from frequency counts as
    # it has no information in our images
    filtered[0] = 0

    cdf = np.zeros(shape=(len(filtered),), dtype=np.uint64)
    cdf[0] = filtered[0]
    for i in range(1, len(filtered)):
      cdf[i] = cdf[i - 1] + filtered[i]

    total = cdf[-1]

    if total == 0:
      return (0, 0)

    lower = 0
    for i, val in enumerate(cdf):
      if float(val) / float(total) > lowerfract:
        break
      lower = i

    upper = 0
    for i, val in enumerate(cdf):
      if float(val) / float(total) > upperfract:
        break
      upper = i

    return (lower, upper)

  def fetch_z_levels(self):
    bounds = Bbox(self.offset, self.shape[:3] + self.offset)

    levelfilenames = [
      'levels/{}/{}'.format(self.mip, z) \
      for z in range(bounds.minpt.z, bounds.maxpt.z)
    ]
    
    with Storage(self.levels_path) as stor:
      levels = stor.get_files(levelfilenames)

    errors = [ 
      level['filename'] \
      for level in levels if level['content'] == None
    ]

    if len(errors):
      raise Exception(", ".join(
          errors) + " were not defined. Did you run a LuminanceLevelsTask for these slices?")

    levels = [(
      int(os.path.basename(item['filename'])),
      json.loads(item['content'].decode('utf-8'))
    ) for item in levels ]

    levels.sort(key=lambda x: x[0])
    levels = [x[1] for x in levels]
    return [ np.array(x['levels'], dtype=np.uint64) for x in levels ]
