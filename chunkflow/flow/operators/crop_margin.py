from chunkflow.chunk import Chunk
from .base import OperatorBase


class CropMarginOperator(OperatorBase):
    """
    crop chunk margin. The margin size should be the same 
    in both side in each dimension.
    """

    def __init__(self,
                 margin_size=None,
                 verbose: bool = True,
                 name: str = 'crop-margin'):
        super().__init__(name=name, verbose=verbose)
        self.margin_size = margin_size

    def __call__(self, chunk, output_bbox=None):
        margin_size = self.margin_size
        if margin_size:
            assert chunk.ndim == len(margin_size)
            if chunk.ndim == 3:
                chunk = chunk[margin_size[0]:chunk.shape[0] -
                              margin_size[0], margin_size[1]:chunk.shape[1] -
                              margin_size[1], margin_size[2]:chunk.shape[2] -
                              margin_size[2]]
            elif chunk.ndim == 4:
                chunk = chunk[margin_size[0]:chunk.shape[0] -
                              margin_size[0], margin_size[1]:chunk.shape[1] -
                              margin_size[1], margin_size[2]:chunk.shape[2] -
                              margin_size[2], margin_size[3]:chunk.shape[3] -
                              margin_size[3]]
            else:
                raise ValueError('the array dimension can only by 3 or 4.')
            global_offset = tuple(
                o + m for o, m in zip(chunk.global_offset, margin_size))
            return Chunk(chunk, global_offset=global_offset)
        else:
            if self.verbose:
                print('automatically crop the chunk to output bounding box.')
            return chunk.cutout(output_bbox.to_slices())
