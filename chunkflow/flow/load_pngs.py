from os import path
import logging

import numpy as np

from chunkflow.lib.bounding_boxes import BoundingBox, Cartesian
from chunkflow.chunk import Chunk

from tqdm import tqdm
import pyspng

def read_png_images(path_prefix: str, bbox: BoundingBox, 
                        volume_offset: tuple = (0, 0, 0),
                        voxel_size: tuple = (1, 1, 1),
                        digit_num: int = 5,
                        dtype: np.dtype = np.uint8):
    
    chunk = Chunk.from_bbox(
        bbox, dtype=dtype, 
        pattern='zero', 
        voxel_size=voxel_size
    )
    assert len(bbox.minpt) == 3
    assert len(volume_offset) == 3

    for z in tqdm( range(bbox.minpt[0], bbox.maxpt[0]) ):
        file_name = f'{path_prefix}{z:0>{digit_num}d}.png'
        file_name = path.expanduser(file_name)
        if path.exists(file_name):
            with open(file_name, "rb") as f:
                img = pyspng.load(f.read())
            img = np.expand_dims(img, axis=0)
            img_chunk = Chunk(
                img,
                voxel_offset=(
                  z+volume_offset[0], volume_offset[1], volume_offset[2]),
                voxel_size=voxel_size
            )
            chunk.blend(img_chunk)
        else:
            logging.warning(f'image file do not exist: {file_name}')
    
    return chunk