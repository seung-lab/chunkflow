import numpy as np
from chunkflow.chunk import Chunk

from .base import OperatorBase


class CreateChunkOperator(OperatorBase):
    def __init__(self, name: str = 'create-chunk', verbose: bool = True):
        super().__init__(name=name, verbose=verbose)

    def __call__(self,
                 size=(64, 64, 64),
                 dtype='uint8',
                 voxel_offset=(0, 0, 0)):
        def _make_black_box():
            slices = tuple(slice(s // 4, -s // 4) for s in size)
            chunk[slices] = 0

        if np.issubdtype(dtype, np.floating):
            chunk = np.random.rand(*size).astype(dtype)
            a = np.zeros(size, dtype=np.uint8)
            ix, iy, iz = np.meshgrid(
                *[np.linspace(0, 1, n) for n in a.shape[1:]], indexing='ij')
            a[:, :, :] = np.abs(np.sin(4 * (ix + iy))) * 255
            _make_black_box()
            return Chunk(chunk, global_offset=voxel_offset)
        elif dtype == 'uint8':
            chunk = np.random.randint(0, 256, size=size, dtype=dtype)
            _make_black_box()
            return Chunk(chunk, global_offset=voxel_offset)
        elif np.issubdtype(dtype, np.integer):
            raise NotImplementedError()
        else:
            raise NotImplementedError()
