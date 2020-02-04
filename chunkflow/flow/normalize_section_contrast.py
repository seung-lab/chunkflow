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
                 lower_clip_fraction: float = 0.01,
                 upper_clip_fraction: float = 0.99,
                 minval: int = 0,
                 maxval: int = np.iinfo(np.uint8).max,
                 name: str = 'normalize-contrast',
                 verbose: bool = True):
        """
        levels_path: (str) path of section histogram files.
        mip: (int) the mip level of section histogram.
        clip_fraction: (float) the fraction of intensity to be clamped.
        minval: (float)
        maxval: (float) Note that the original algorithm use default maxval as 
                float(np.iinfo(np.float32)).max. It is meaningless to use value 
                larger than the max value of uint8.
        """
        super().__init__(name=name, verbose=verbose)
        assert 0 <= lower_clip_fraction <= 1
        assert 0 <= upper_clip_fraction <= 1
        assert lower_clip_fraction < upper_clip_fraction

        self.levels_path = levels_path
        self.mip = int(mip)
        self.lower_clip_fraction = float(lower_clip_fraction)
        self.upper_clip_fraction = float(upper_clip_fraction)

        self.minval = minval
        self.maxval = maxval
       
        # intensity value lookup table cache
        self.lookup_tables = dict()
        self.stor = Storage(self.levels_path)

    def __call__(self, chunk):
        # this is a image, not affinitymap
        assert chunk.ndim == 3
        assert chunk.dtype is np.uint8
        
        for z in range(chunk.bbox.minpt.z, chunk.bbox.maxpt.z):
            lookup_table = self.fetch_lookup_table(z)
            slices = tuple(slice(z, z+1), *chunk.slices[-2:])
            image = chunk.cutout(slices)
            image = lookup_table[image]
            chunk.save(image)

        return chunk

    def find_section_clamping_values(self, zlevel):
        """compute the clamping values for each section."""
        # remove the np.copy from original code since we only need this once
        filtered = zlevel

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
            if float(val) / float(total) > self.lower_clip_fraction:
                break
            lower = i

        upper = 0
        for i, val in enumerate(cdf):
            if float(val) / float(total) > self.upper_clip_fraction:
                break
            upper = i

        return (lower, upper)

    def fetch_lookup_table(self, z):
        """
        readout the histograms in each corresponding section.
        TODO: use local cache for the z levels
        """
        if z not in self.lookup_tables:
            levelfilename = f'levels/{self.mip}/{z}'
            data = self.stor.get_files(levelfilename)
            data = json.loads(data['content'].decode('utf-8'))
            levels = np.array(data['levels'], dtype=np.uint64)
            lower, upper = self.find_section_clamping_values(levels)
            
            if lower == upper:
                lookup_table = np.arange(0, 256, dtype=np.uint8)
            else:
                # compute the lookup table
                lookup_table = np.arange(0, 256, dtype=np.float32)
                lookup_table = (lookup_table - float(lower)) * (
                    self.maxval / (float(upper) - float(lower)))
                lookup_table = np.round(lookup_table)
                lookup_table = lookup_table.astype( np.uint8 ) 

            lookup_table = np.clip(lookup_table, self.minval, self.maxval)
            self.lookup_tables[z] = lookup_table
        return self.lookup_tables[z]
