__doc__ = """Image chunk class"""

import numpy as np
from chunkflow.chunk import Chunk
from waterz import evaluate


class Segmentation(Chunk):
    """
    a chunk of segmentation volume.
    """

    def __new__(cls, array, **kwargs):
        if 'global_offset' in kwargs:
            global_offset = kwargs['global_offset']
        elif isinstance(array, Chunk):
            global_offset = array.global_offset
        else:
            global_offset = None

        obj = Chunk(array, global_offset=global_offset, *kwargs).view(cls)
        return obj

    def compare(self, other):
        if not np.issubdtype(self.dtype, np.uint64):
            this = self.astype(np.uint64)
        else:
            this = self

        if not np.issubdtype(other.dtype, np.uint64):
            other = other.astype(np.uint64)

        return evaluate(this, other)
