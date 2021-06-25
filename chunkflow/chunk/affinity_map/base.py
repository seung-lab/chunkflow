__doc__ = """Image chunk class"""

import numpy as np

from chunkflow.chunk import Chunk

class AffinityMap(Chunk):
    """
    a chunk of affinity map. It has x,y,z three channels with single precision.
    """
    def __init__(self, array, voxel_offset=None):
        assert array.ndim == 4
        assert np.issubdtype(array.dtype, np.float32)
        assert array.shape[0] == 3
        super().__init__(array, voxel_offset=voxel_offset)

    def quantize(self):
        # only use the last channel, it is the Z affinity
        # if this is affinitymap
        image = self[-1, :, :, :]
        image = (image * 255).astype(np.uint8)
        image = Chunk(image, voxel_offset=self.voxel_offset, voxel_size=self.voxel_size)
        return image