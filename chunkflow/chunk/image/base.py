__doc__ = """Image chunk class"""

import numpy as np

from chunkflow.chunk import Chunk
from .adjust_grey import normalize_section_shang
from .convnet.inferencer import Inferencer


class Image(Chunk):
    """
    a chunk of image volume.
    """
    def __init__(self, array: np.ndarray, global_offset=None):
        super().__init__(array, global_offset=global_offset)

    def inference(self, inferencer: Inferencer):
        """run convolutional net inference for this image chunk"""
        return inferencer(self)

    def normalize_section_shang(self, nominalmin, nominalmax, clipvalues):
        return normalize_section_shang(self.array, nominalmin, nominalmax,
                                       clipvalues)
