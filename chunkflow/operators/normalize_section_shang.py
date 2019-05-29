# -*- coding: utf-8 -*-
import numpy as np

from .operator_base import OperatorBase
from .adjust_grey import normalize as grey_normalize


class NormalizeSectionShangOperator(OperatorBase):
    """Contrast Correction based on section min/max within the chunk, Shang's method.
    """
    def __init__(self,
                 nominalmin: float=None,
                 nominalmax: float=None,
                 clipvalues: bool=False,
                 name: str='normalize-gray', verbose=False):
        """
        nominalmin/max: (float)
        clipvalues: (bool)
        """
        super().__init__(name=name, verbose=verbose)
        assert nominalmin < nominalmax

        self.nominalmin = nominalmin
        self.nominalmax = nominalmax
        self.clipvalues = clipvalues

    def __call__(self, chunk, debug=False):
        # this is an image chunk, not affinitymap
        assert chunk.ndim == 3
        originaltype = chunk.dtype
        chunk = chunk.astype(np.float32)
        if debug:
            print(chunk.shape)

        # number of bits per voxel
        nbits = np.dtype(originaltype).itemsize * 8
        nominalmax = float(2**nbits - 1)

        nominalmin = self.nominalmin if self.nominalmin is not None else 0.0
        nominalmax = self.nominalmax if self.nominalmax is not None else nominalmax

        normalization = 'fill'
        
        # stack/chunk-wise normalization first if necessary (for blank slices within a valid stack)
        #chunk = grey_normalize(chunk, normalization, target_scale = [-1,1], min_max_invalid = [True]*2, make_copy=False)
        
        # slice-wise normalization
        # Note in chunkflow the first dim is z/slice
        for ii in range(chunk.shape[0]):
            grey_normalize(chunk[ii,:,:], normalization, target_scale = [nominalmin,nominalmax], 
                    min_max_invalid = [True,True], do_clipping = self.clipvalues, make_copy=False)

        # cast to original data type if necessary
        #chunk = np.round(chunk)
        #chunk = chunk.astype(originaltype)

        return chunk
