__doc__ = """Image chunk class"""
import numpy as np
from chunkflow.chunk import Chunk

class AffinityMap(Chunk):
    """
    a chunk of affinity map. It has x,y,z three channels with single precision.
    """
    def __init__(self, array, global_offset=None):
        super().__init__(array, global_offset=global_offset)

    def quantize(self):
        # only use the last channel, it is the Z affinity
        # if this is affinitymap
        image = self[-1, :, :, :]
        image = (image * 255).astype(np.uint8)
        return image
