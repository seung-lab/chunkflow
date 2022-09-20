from typing import Union

import numpy as np

from cloudvolume import CloudVolume
from .lib.cartesian_coordinate import BoundingBox, Cartesian
from .chunk import Chunk


class Volume:
    """The major difference with CloudVolume is that we use C order here. 
    ZYX indexing.

    Args:
        CloudVolume (class): the cloud-volume class
    """
    def __init__(self, vol: CloudVolume) -> None:
        self.vol = vol

    @property
    def dtype(self):
        return self.vol.dtype

    @classmethod
    def from_cloudvolume_path(cls, path: str, *arg, **kwargs):
        vol = CloudVolume(path, *arg, **kwargs)
        return cls(vol)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, vol_path: str):
        vol = CloudVolume.from_numpy(np.transpose(arr), vol_path=vol_path)
        return cls(vol)

    def cutout(self, key: Union[BoundingBox, list]):
        if isinstance(key, BoundingBox):
            arr = self.vol[ key.to_slices()[::-1] ]
            voxel_offset = key.start
        elif isinstance(key, list):
            arr = self.vol[key[::-1]]
            voxel_offset = Cartesian(key[0].start, key[1].start, key[2].start)
        else:
            raise ValueError('we only support BoundingBox or a list of slices')

        # transpose
        arr = np.transpose(arr)
        arr = np.asarray(arr)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        chunk = Chunk(arr, voxel_offset=voxel_offset) 
        return chunk

    def _auto_convert_dtype(self, chunk: Chunk):
        """convert the data type to fit volume datatype"""
        if np.issubdtype(self.dtype, np.floating) and \
                np.issubdtype(chunk.dtype, np.uint8):
            chunk = chunk.astype(self.dtype)
            chunk /= 255.
            # chunk = chunk / chunk.array.max() * np.iinfo(volume.dtype).max
        elif np.issubdtype(self.dtype, np.uint8) and \
                np.issubdtype(chunk.dtype, np.floating):
            chunk.max() <= 1.
            chunk *= 255

        if self.dtype != chunk.dtype:
            print(f'converting chunk data type {chunk.dtype} ' + 
                         f'to volume data type: {self.dtype}')
            return chunk.astype(self.dtype)
        else:
            return chunk

    def save(self, chunk: Chunk):
        chunk = self._auto_convert_dtype(chunk)
        
        # transpose czyx to xyzc order
        arr = np.transpose(chunk.array)
        self[chunk.slices[::-1]] = arr