from os import path
import logging

import numpy as np
from cloudvolume.lib import Bbox

from chunkflow.chunk import Chunk

from tqdm import tqdm
from PIL import Image
# the default setting have a limit to avoid decompression bombs
# https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=maximum_pixels#PIL.Image.open
Image.MAX_IMAGE_PIXELS = None 


def read_png_images(path_prefix: str, bbox: Bbox, 
                        volume_offset: tuple = (0, 0, 0),
                        voxel_size: tuple = (1, 1, 1),
                        dtype: np.dtype = np.uint8):
    
    chunk = Chunk.from_bbox(
        bbox, dtype=dtype, 
        all_zero=True, 
        voxel_size=voxel_size
    )
    assert len(bbox.minpt) == 3
    assert len(volume_offset) == 3

    for z in tqdm( range(bbox.minpt[0], bbox.maxpt[0]) ):
        file_name = '{}{:0>5d}.png'.format(path_prefix, z)
        file_name = path.expanduser(file_name)
        if path.exists(file_name):
            img = Image.open(file_name)
            img = np.asarray(img)
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