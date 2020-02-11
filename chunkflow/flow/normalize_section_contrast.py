# -*- coding: utf-8 -*-
import json
import numpy as np
from cloudvolume.storage import SimpleStorage
from chunkflow.chunk import Chunk

from .base import OperatorBase


class NormalizeSectionContrastOperator(OperatorBase):
    """Contrast Correction based on LuminanceLevelsTask output.

    Note that this operator was modified from Will's ContrastNormalizationTask in igneous:
    https://github.com/seung-lab/igneous/blob/master/igneous/tasks.py#L735
    """
    def __init__(self,
                 levels_path: str,
                 lower_clip_fraction: float = 0.01,
                 upper_clip_fraction: float = 0.01,
                 minval: int = 1,
                 maxval: int = np.iinfo(np.uint8).max,
                 #maxval: int = np.iinfo(np.uint32).max,
                 name: str = 'normalize-contrast',
                 verbose: bool = True):
        """
        levels_path: (str) path of section histogram files.
        clip_fraction: (float) the fraction of intensity to be clamped.
        minval: (float)
        maxval: (float) Note that the original algorithm use default maxval as 
                float(np.iinfo(np.float32)).max. It is meaningless to use value 
                larger than the max value of uint8.
        """
        super().__init__(name=name, verbose=verbose)
        assert 0 <= lower_clip_fraction <= 1
        assert 0 <= upper_clip_fraction <= 1
        assert lower_clip_fraction + upper_clip_fraction <= 1

        self.levels_path = levels_path
        self.lower_clip_fraction = float(lower_clip_fraction)
        self.upper_clip_fraction = float(upper_clip_fraction)

        self.minval = minval
        self.maxval = maxval
       
        # intensity value lookup table cache
        self.lookup_tables = dict()
        self.stor = SimpleStorage(self.levels_path)
        self.verbose = verbose

    def __call__(self, chunk):
        # this is a image, not affinitymap
        assert chunk.ndim == 3
        # we have to use np.dtype function to make it match
        # https://stackoverflow.com/questions/26921836/correct-way-to-test-for-numpy-dtype
        assert chunk.dtype is np.dtype(np.uint8)
        
        for z in range(chunk.bbox.minpt[-3], chunk.bbox.maxpt[-3]):
            lookup_table = self.fetch_lookup_table(z)
            slices = (slice(z, z+1), *chunk.slices[-2:])
            image = chunk.cutout(slices)
            image_global_offset = image.global_offset
            image = lookup_table[image]
            image = Chunk(image, global_offset=image_global_offset)
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
            if float(val) / float(total) > 1 - self.upper_clip_fraction:
                break
            upper = i
        
        return lower, upper

    def fetch_lookup_table(self, z):
        """
        readout the histograms in each corresponding section.
        lookup tables are constructed and cached.
        """
        if z not in self.lookup_tables:
            data = self.stor.get_file(f'{z}')
            assert data is not None
            data = json.loads(data.decode('utf-8'))
            levels = np.array(data['levels'], dtype=np.uint64)
            lower, upper = self.find_section_clamping_values(levels)
             
            if lower == upper:
                lookup_table = np.arange(0, 256, dtype=np.uint8)
            else:
                # compute the lookup table
                lookup_table = np.arange(0, 256, dtype=np.float32)
                lookup_table = (lookup_table - float(lower)) * (
                    self.maxval / (float(upper) - float(lower)))
            np.clip(lookup_table, self.minval, self.maxval, out=lookup_table)
            lookup_table = np.round(lookup_table)
            lookup_table = lookup_table.astype( np.uint8 ) 
            
            self.lookup_tables[z] = lookup_table
        return self.lookup_tables[z]
