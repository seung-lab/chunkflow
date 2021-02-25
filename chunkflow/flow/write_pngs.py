from .base import OperatorBase

from warnings import warn
import os
import numpy as np

from tqdm import tqdm
from skimage.io import imsave

from chunkflow.chunk import Chunk


class WritePNGsOperator(OperatorBase):
    def __init__(self,
                 output_path: str = './pngs/',
                 name: str = 'save-pngs'):
        super().__init__(name=name)

        if not os.path.isdir(output_path):
            warn('output path do not exist, will create %s.' % output_path)
            os.makedirs(output_path)

        self.output_path = output_path

    def __call__(self, chunk: Chunk):
        assert isinstance(chunk, Chunk)
        assert chunk.ndim == 3
        for z in tqdm(range(chunk.voxel_offset[0], chunk.bbox.maxpt[0])):
            img = chunk.cutout((slice(z,z+1), chunk.slices[1], chunk.slices[2]))
            img = img.array[0,:,:]
            imsave(os.path.join(self.output_path, '{:05d}.png'.format(z)), img)