# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from cloudvolume import CloudVolume, Storage
from cloudvolume.lib import Vec, Bbox

from .base import OperatorBase


class NormalizeSectionContrastOperator(OperatorBase):
    """Contrast Correction based on LuminanceLevelsTask output.

    Note that this operator was modified from Will's ContrastNormalizationTask in igneous:
    https://github.com/seung-lab/igneous/blob/master/igneous/tasks.py#L735
    """

    def __init__(self,
                 levels_path: str,
                 mip: int,
                 clip_fraction: float,
                 minval: float = None,
                 maxval: float = None,
                 name: str = 'normalize-contrast',
                 verbose: bool = True):
        """
        levels_path: (str) path of section histogram files.
        mip: (int) the mip level of section histogram.
        clip_fraction: (float) the fraction of intensity to be clamped.
        minval: (float)
        """
        super().__init__(name=name, verbose=verbose)
        assert 0 <= clip_fraction <= 1

        self.levels_path = levels_path
        self.mip = int(mip)
        self.clip_fraction = float(clip_fraction)
        self.minval = minval
        self.maxval = maxval

    def __call__(self, chunk):
        """
        TODO: implement using precomputed lookup tables.
        The intensity value map could be precomputed as a list, 
        and we only need to replace the voxel intensity according to the list.
        """
        # this is a image, not affinitymap
        assert chunk.ndim == 3
        image = np.transpose(chunk).astype(np.float32)

        # translate to xyz order since cloudvolume is working this F order
        offset = Vec(*chunk.global_offset[::-1])
        shape = Vec(*image.shape)
        bounds = Bbox.from_delta(offset, shape)

        zlevels = self.fetch_z_levels(bounds)

        # number of bits per voxel
        nbits = np.dtype(chunk.dtype).itemsize * 8
        maxval = float(2**nbits - 1)

        for z in range(bounds.minpt.z, bounds.maxpt.z):
            imagez = z - bounds.minpt.z
            zlevel = zlevels[imagez]
            (lower, upper) = self.find_section_clamping_values(
                zlevel, self.clip_fraction, 1 - self.clip_fraction)
            if lower == upper:
                continue
            img = image[:, :, imagez]
            img = (img - float(lower)) * (maxval /
                                          (float(upper) - float(lower)))
            image[:, :, imagez] = img

        image = np.round(image)

        minval = self.minval if self.minval is not None else 0.0
        maxval = self.maxval if self.maxval is not None else maxval

        image = np.clip(image, minval, maxval)

        chunk = np.transpose(image).astype(chunk.dtype)
        chunk = Chunk(chunk, global_offset=(*offset, )[::-1])
        return chunk

    def find_section_clamping_values(self, zlevel, lowerfract, upperfract):
        """compute the clamping values for each section."""
        filtered = np.copy(zlevel)

        # remove pure black from frequency counts as
        # it has no information in our images
        filtered[0] = 0

        cdf = np.zeros(shape=(len(filtered), ), dtype=np.uint64)
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

    def fetch_z_levels(self, bounds):
        """
        readout the histograms in each corresponding section.
        TODO: use local cache for the z levels
        """
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
            raise Exception(
                ", ".join(errors) +
                " were not defined. Did you run a LuminanceLevelsTask for these slices?"
            )

        levels = [(int(os.path.basename(item['filename'])),
                   json.loads(item['content'].decode('utf-8')))
                  for item in levels]

        levels.sort(key=lambda x: x[0])
        levels = [x[1] for x in levels]
        return [np.array(x['levels'], dtype=np.uint64) for x in levels]
