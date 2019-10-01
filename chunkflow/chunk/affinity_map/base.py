__doc__ = """Image chunk class"""
import numpy as np
from chunkflow.chunk import Chunk

class AffinityMap(Chunk):
    """
    a chunk of affinity map. It has x,y,z three channels with single precision.
    """
    def __new__(cls, array, **kwargs):
        if 'global_offset' in kwargs:
            global_offset = kwargs['global_offset']
        elif isinstance(array, Chunk):
            global_offset = array.global_offset
        else:
            global_offset = None

        assert np.issubdtype(array.dtype, np.floating)
        assert 4 == array.ndim
        assert 3 == array.shape[0] 
        obj = Chunk(array, global_offset=global_offset, *kwargs).view(cls)
        return obj

    def quantize(self):
        # only use the last channel, it is the Z affinity
        # if this is affinitymap
        image = self[-1, :, :, :]
        image = (image * 255).astype(np.uint8)
        return image
        



