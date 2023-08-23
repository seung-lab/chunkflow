import numpy as np

from chunkflow.chunk import Chunk
import nrrd


def execute(file_name: str=None, voxel_offset: tuple=None,
        voxel_size: tuple=None, dtype: str=None, layer_type: str=None):
    """Read NRRD file."""
    arr, _ = nrrd.read(file_name)
    
    if dtype:
        arr = arr.astype(dtype)
    
    chunk = Chunk(arr, voxel_offset=voxel_offset, voxel_size=voxel_size)
    breakpoint()
    return chunk
