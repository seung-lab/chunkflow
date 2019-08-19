# -*- coding: utf-8 -*-
import numpy as np

from .base import OperatorBase
from chunkflow.chunk.image import Image


class NormalizeSectionShangOperator(OperatorBase):
    """Contrast Correction based on section min/max within the chunk, 
    Shang's method.
    """
    def __init__(self,
                 nominalmin: float = None,
                 nominalmax: float = None,
                 clipvalues: bool = False,
                 name: str = 'normalize-gray',
                 verbose=False):
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
        if debug:
            print(chunk.shape)

        image = Image(chunk)
        return image.normalize_section_shang(self.nominalmin, self.nominalmax,
                                             self.clipvalues)
