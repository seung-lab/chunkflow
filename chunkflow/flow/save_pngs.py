from .base import OperatorBase

from warnings import warn
import os
import numpy as np

from tqdm import tqdm
import pyspng
from PIL import Image

from chunkflow.chunk import Chunk


class SavePNGsOperator(OperatorBase):
    def __init__(self,
                 output_path: str = './pngs/',
                 dtype: str = 'uint8',
                 name: str = 'save-pngs'):
        super().__init__(name=name)

        if not os.path.isdir(output_path):
            warn(f'output path do not exist, will create %s {output_path}')
            os.makedirs(output_path)

        self.dtype = np.dtype(dtype)
        self.output_path = output_path

    def __call__(self, chunk: Chunk):
        assert isinstance(chunk, Chunk)
        if not np.issubdtype(chunk.dtype, self.dtype):
            chunk = chunk.astype(self.dtype)

        if chunk.is_affinity_map:
            properties = chunk.properties
            chunk = (chunk[1,...] + chunk[2,...]) / 2. * 255.
            chunk = chunk.astype(np.uint8)
            chunk = Chunk(chunk)
            chunk.set_properties(properties)

        assert chunk.ndim == 3
        for z in tqdm(range(chunk.voxel_offset[0], chunk.bbox.maxpt[0])):
            img = chunk.cutout((slice(z,z+1), chunk.slices[1], chunk.slices[2]))
            img = img.array[0,:,:]
            filename = os.path.join(self.output_path, f"{z:05d}.png")
            if np.issubdtype(img.dtype, np.uint16):
                # pyspng do not support 16 bit image well.
                img = Image.fromarray(img)
                img.save(filename)
            else:
                binary = pyspng.encode(
                    img,
                    progressive=pyspng.ProgressiveMode.PROGRESSIVE,
                    compress_level=6,
                )
                with open(filename, "wb") as fout:
                    fout.write(binary)