from __future__ import annotations
from typing import Union
from abc import ABC, abstractmethod, abstractproperty
from functools import cached_property

import numpy as np

from cloudvolume import CloudVolume
from .lib.cartesian_coordinate import BoundingBox, Cartesian
from .chunk import Chunk

class AbstractVolume(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractproperty
    def bounding_box(self) -> BoundingBox:
        pass

    @abstractproperty
    def block_size(self) -> Cartesian:
        pass

    @abstractproperty
    def voxel_size(self) -> Cartesian:
        pass

    @abstractmethod
    def cutout(self, key: Union[BoundingBox, list]) -> Chunk:
        pass

    @abstractmethod
    def save(self, chk: Chunk) -> None:
        pass

class PrecomputedVolume(AbstractVolume):
    """The major difference with CloudVolume is that we use C order here. 
    ZYX indexing.

    Args:
        CloudVolume (class): the cloud-volume class
    """
    def __init__(self, vol: CloudVolume) -> None:
        self.vol = vol
    
    @classmethod
    def from_cloudvolume_path(cls, path: str, *arg, **kwargs) -> PrecomputedVolume:
        vol = CloudVolume(path, *arg, fill_missing=True, **kwargs)
        return cls(vol)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, vol_path: str) -> PrecomputedVolume:
        vol = CloudVolume.from_numpy(np.transpose(arr), vol_path=vol_path)
        return cls(vol)

    @cached_property
    def bounding_box(self):
        bbox = self.vol.bounds
        bbox = BoundingBox.from_bbox(bbox)
        # from xyz to zyx
        bbox.inverse_order()
        return bbox
    
    @cached_property
    def start(self) -> Cartesian:
        self.bounding_box.start

    @cached_property
    def stop(self) -> Cartesian:
        self.bounding_box.stop

    @cached_property
    def voxel_size(self) -> Cartesian:
        voxel_size = self.vol.resolution[::-1]
        voxel_size = Cartesian.from_collection(voxel_size)
        return voxel_size

    @cached_property
    def dtype(self):
        return self.vol.dtype

    @cached_property
    def block_size(self):
        return Cartesian.from_collection(
            self.vol.chunk_size[::-1])

    @cached_property
    def block_bounding_boxes(self) -> List[BoundingBox]:
        bboxes = []
        for z in range(self.start.z, self.stop.z-self.block_size.z, self.block_size.z):
            for y in range(self.start.y, self.stop.y-self.block_size.y, self.block_size.y):
                for x in range(self.start.x, self.stop.x-self.block_size.x, self.block_size.x):
                    bbox = BoundingBox.from_delta(Cartesian(z,y,x), self.block_size)
                    bboxes.append(bbox)
        return bboxes

    @property
    def nonzero_block_bounding_boxes(self) -> List[BoundingBox]:
        nnz_bboxes = []
        for bbox in self.block_bounding_boxes:
            chunk = self.cutout(bbox)
            if np.all(chunk > 0):
                nnz_bboxes.append(bbox)

        return nnz_bboxes
   
    @cached_property
    def shape(self):
        return self.vol.shape

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


# class ZarrVolume(AbstractVolume):

# class SynapseVolume:
