import numpy as np

from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
from chunkflow.chunk import Chunk

import mrcfile

def execute(bbox: BoundingBox, fname: str = None):

    with mrcfile.mmap(fname) as mrc:
        # print(mrc.header)
        # print(f'volume shape: {mrc.data.shape}')
        # print(f'voxel size: {mrc.voxel_size}')
        voxel_size = mrc.voxel_size
        # voxel_size = [int(x/10.) for x in mrc.voxel_size]
        # voxel_size = Cartesian.from_collection(voxel_size)
        # print(f'voxel size: {voxel_size}')
        arr = mrc.data[bbox.to_slices()]

    # somehow, mrcfile pick it up with int8!
    arr = arr.view(np.uint8)
    z = int(voxel_size.z / 10)
    y = int(voxel_size.y / 10)
    x = int(voxel_size.x / 10)
    voxel_size = Cartesian(z, y, x)
    chunk = Chunk(arr, voxel_offset=bbox.start, voxel_size=voxel_size)
    return chunk 
