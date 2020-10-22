import numpy as np
from os import path
from cloudvolume import Bbox
from chunkflow.chunk import Chunk


def read_png_images(path_prefix: str, bbox: Bbox, dtype: np.dtype = np.uint8):
    # soft dependency
    from PIL import Image 
    chunk = Chunk.from_bbox(bbox, dtype=dtype, all_zero=True)
    assert len(bbox.minpt) == 3
    
    for z in range(bbox.minpt[0], bbox.maxpt[0]):
        file_name = '{}{:0>5d}.png'.format(path_prefix, z)
        file_name = path.expanduser(file_name)
        img = Image.open(file_name)
        img_chunk = Chunk(img, voxel_offset=(z, 0, 0))
        chunk.blend(img_chunk)
    
    return chunk