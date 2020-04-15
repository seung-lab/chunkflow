__doc__ = """Image chunk class"""

import multiprocessing

import numpy as np
from .base import Chunk

from waterz import evaluate
import kimimaro


class Segmentation(Chunk):
    """
    a chunk of segmentation volume.
    """
    def __init__(self, array, global_offset=None):
        super().__init__(array, global_offset=global_offset)
        assert array.ndim == 3
        assert np.issubdtype(array.dtype, np.integer)

    @classmethod
    def from_chunk(cls, chunk):
        assert isinstance(chunk, Chunk)
        return cls(chunk.array, global_offset=chunk.global_offset)

    def evaluate(self, groundtruth):
        if not np.issubdtype(self.dtype, np.uint64):
            this = self.astype(np.uint64)
        else:
            this = self

        if not np.issubdtype(groundtruth.dtype, np.uint64):
            groundtruth = groundtruth.astype(np.uint64)
        
        if isinstance(groundtruth, Chunk):
            groundtruth = groundtruth.array

        return evaluate(this.array, groundtruth)

    def skeletonize(self, voxel_size):
        skels = kimimaro.skeletonize(
            self.array,
            anisotropy=voxel_size,
            parallel=multiprocessing.cpu_count() // 2
        )
        return skels

