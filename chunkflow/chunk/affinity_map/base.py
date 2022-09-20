__doc__ = """Image chunk class"""

import numpy as np

from chunkflow.lib.cartesian_coordinate import Cartesian
from chunkflow.chunk import Chunk

class AffinityMap(Chunk):
    """
    a chunk of affinity map. It has x,y,z three channels with single precision.
    """
    def __init__(self, array, 
            voxel_offset: Cartesian=None, 
            voxel_size: Cartesian=None ):
        assert array.ndim == 4
        assert np.issubdtype(array.dtype, np.float32)
        assert array.shape[0] == 3
        super().__init__(array, voxel_offset=voxel_offset, voxel_size=voxel_size)

    def quantize(self, mode: str='xy'):
        """transform affinity map to gray scale image

        Args:
            mode (str, optional): tranformation mode. Defaults to 'xy'.

        Raises:
            ValueError: only support mode of xy and z.

        Returns:
            Chunk: the gray scale image chunk
        """
        if mode == 'z':
            # only use the last channel, it is the Z affinity
            # if this is affinitymap
            image = self[-1, :, :, :]
        elif mode == 'xy':
            image = (self[0,...] + self[1, ...]) / 2.
        else:
            raise ValueError(f'only support xy and z mode, but got {mode}')

        image = (image * 255.).astype(np.uint8)
        image = Chunk(image)
        image.set_properties(self.properties)
        # assert np.issubdtype(image.dtype, np.uint8)
        return image