import os


import numpy as np

from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
from chunkflow.chunk import Chunk

from tqdm import tqdm
import pyspng


def load_png_image(file_name: str):
    with open(file_name, "rb") as f:
        arr = pyspng.load(f.read())
    if np.ndim(arr) == 3:
        arr = arr[:, :, 0]
    return arr


def load_png_images(
        path_prefix: str, 
        bbox: BoundingBox = None, 
        voxel_offset: Cartesian = Cartesian(0, 0, 0),
        voxel_size: Cartesian = Cartesian(1, 1, 1),
        digit_num: int = 5,
        dtype: np.dtype = None):
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)
        
    file_names = []

    if bbox is None:
        if os.path.isfile(path_prefix):
            file_names.append(path_prefix)
        else:
            if os.path.isdir(path_prefix):
                dir_path = path_prefix
            else:
                dir_path = os.path.dirname(path_prefix)
            fname = os.path.expanduser(dir_path)
            for fname in sorted(os.listdir(dir_path)):
                if fname.endswith('.png'):
                    fname = os.path.join(dir_path, fname)
                    file_names.append(fname)
        arr = load_png_image(file_names[0])
        shape = Cartesian(len(file_names), arr.shape[0], arr.shape[1])
        if dtype is None:
            dtype = arr.dtype
        bbox = BoundingBox.from_delta(voxel_offset, shape)
    else:
        for z in range(bbox.start[0], bbox.stop[0]):
            file_name = f'{path_prefix}{z:0>{digit_num}d}.png'
            file_name = os.path.expanduser(file_name)
            file_names.append(file_name)

    chunk = Chunk.from_bbox(
        bbox, dtype=dtype, 
        pattern='zero', 
        voxel_size=voxel_size
    )

    for z_offset, file_name in tqdm(enumerate(file_names)):
        if os.path.exists(file_name):
            if z_offset > 0:
                arr = load_png_image(file_name)

            if arr.dtype != dtype:
                arr = arr.astype(dtype)
            # breakpoint()
            chunk.array[z_offset, :, :] = arr[
                bbox.start[1]:bbox.stop[1], 
                bbox.start[2]:bbox.stop[2]]
        else:
            print(f'image file do not exist: {file_name}')
    
    return chunk