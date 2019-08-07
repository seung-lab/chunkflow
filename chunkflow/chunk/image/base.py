__doc__ = """Image chunk class"""

from chunkflow.chunk import Chunk
from .adjust_grey import normalize_section_shang
from .convnet_inference.engine import Engine


class Image(Chunk):
    """
    a chunk of image volume.
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

    def inference(self, engine: Engine):
        """run convolutional net inference for this image chunk"""
        return engine(self)

    def normalize_section_shang(self, nominalmin, nominalmax, clipvalues):
        return normalize_section_shang(self, nominalmin, nominalmax,
                                       clipvalues)
