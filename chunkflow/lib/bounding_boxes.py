# support the class method with parameter type of itself
from __future__ import annotations

import logging
import os
from collections import UserList, namedtuple
from math import ceil
from typing import Union
from numbers import Number
import itertools

from copy import deepcopy

import numpy as np
import h5py

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox


class Cartesian(namedtuple('Cartesian', ['z', 'y', 'x'])):
    """Cartesian coordinate or offset."""
    __slots__ = ()
    
    def __sub__(self, offset: Union[Cartesian, int]):
        """subtract to another voxel coordinate

        Args:
            offset (Cartesian, int): another voxel coordinate
        """
        if isinstance(offset, int):
            offset = (offset, offset, offset)
        return Cartesian(*[x-o for x, o in zip(self, offset)])
           
    def __add__(self, offset: Union[Cartesian, int]):
        """add another coordinate

        Args:
            offset (Cartesian, int): offset
        """
        if isinstance(offset, int):
            return Cartesian(*[x+offset for x in self])
        else:
            return Cartesian(*[x+o for x, o in zip(self, offset)])

    def __mul__(self, m: Number) -> Cartesian:
        return Cartesian(*[x*m for x in self])

    def __floordiv__(self, d: int):
        return Cartesian(*[x // d for x in self])

    def __truediv__(self, d: Number):
        return Cartesian(*[x/d for x in self])

    def __mod__(self, d: int):
        return Cartesian(*[x%d for x in self])

    def __lt__(self, other: Cartesian) -> bool:
        if self.x < other.x and self.y < other.y and self.z < other.z:
            return True
        else:
            return False

    def __le__(self, other: Cartesian) -> bool:
        if self.x <= other.x and self.y <= other.y and self.z <= other.z:
            return True
        else:
            return False

    def __gt__(self, other: Cartesian) -> bool:
        if self.x > other.x and self.y > other.y and self.z > other.z:
            return True
        else:
            return False

    def __ge__(self, other: Cartesian) -> bool:
        if self.z >= other.z and self.y >= other.y and self.x >= other.x:
            return True
        else:
            return False

    def __ne__(self, other: Cartesian) -> bool:
        if self.z != other.z and self.y != other.y and self.x != other.x:
            return True
        else:
            return False

    # def __isub__(self, other: Union[Cartesian,Number]) -> Cartesian:



    @property
    def vec(self):
        return Vec(*self)


class BoundingBox(Bbox):
    def __init__(self, 
            minpt: Union[list, Cartesian], 
            maxpt: Union[list, Cartesian], 
            dtype=None, 
            voxel_size: Cartesian = None):
        if isinstance(minpt, Cartesian):
            minpt = minpt.vec
        
        if isinstance(maxpt, Cartesian):
            maxpt = maxpt.vec
        super().__init__(minpt, maxpt, dtype=dtype)
        self._voxel_size = voxel_size

    @classmethod
    def from_bbox(cls, bbox: Bbox, voxel_size: tuple = None):
        return cls(bbox.minpt, bbox.maxpt, voxel_size=voxel_size)

    @classmethod
    def from_delta(cls, minpt, plus):
        bbox = super().from_delta(minpt, plus)
        return cls.from_bbox(bbox)

    @classmethod
    def from_list(cls, x: list):
        bbox = Bbox.from_list(x)
        return cls.from_bbox(bbox)

    @classmethod
    def from_points(cls, x: np.ndarray):
        bbox = Bbox.from_points(x)
        return cls.from_bbox(bbox)

    @classmethod
    def from_center(cls, center: Cartesian, extent: int,
            even_size: bool = True):
        """Create bounding box from center and extent

        Args:
            center (Cartesian): center coordinate
            extent (int): the range to extent, like radius
            even_size (bool): produce even size or odd size including the center.
        """
        minpt = center - extent
        # the maxpt is not inclusive, so we need +1
        maxpt = center + extent
        if not even_size:
            # this will make the size to be odd
            maxpt += 1
        return cls(minpt, maxpt)

    def __repr__(self):
        return f'BoundingBox({self.minpt}, {self.maxpt}, dtype={self.dtype}, voxel_size={self.voxel_size})'

    def clone(self):
        bbox = Bbox(self.minpt, self.maxpt, dtype=self.dtype)
        bbox = bbox.clone()
        return BoundingBox.from_bbox(bbox, voxel_size=self.voxel_size)

    def adjust(self, size: Union[Cartesian, int, tuple, list, Vec]):
        if size is None:
            logging.warn('adjusting bounding box size is None!')
            return self

        if not isinstance(size, int):
            assert len(size)==3 or len(size)==6
            size = Vec(*size)
        self.minpt -= size[:3]
        self.maxpt += size[-3:]
        return self

    def union(self, bbox2):
        """Merge another bounding box

        Args:
            bbox2 (Union[BoundingBox, Bbox]): another bounding box

        Returns:
            BoundingBox: bounding box after merging
        """
        if isinstance(bbox2, BoundingBox):
            assert self.voxel_size == bbox2.voxel_size

        self.minpt = np.minimum(self.minpt, bbox2.minpt)
        self.maxpt = np.maximum(self.maxpt, bbox2.maxpt)
        return self

    def contains(self, point: Union[tuple, Vec, list]):
        assert 3 == len(point)
        return np.all(np.asarray(
            (self.maxpt >= Vec(*point)))) and np.all(
                np.asarray((self.minpt <= Vec(*point)))) 

    @property
    def shape(self):
        return Cartesian(*(self.maxpt - self.minpt))

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def left_neighbors(self):
        sz = self.size3()

        minpt = deepcopy(self.minpt)
        minpt[0] -= sz[0]
        bbox_z = self.from_delta(minpt, sz)

        minpt = deepcopy(self.minpt)
        minpt[1] -= sz[1]
        bbox_y = self.from_delta(minpt, sz) 

        minpt = deepcopy(self.minpt)
        minpt[2] -= sz[2]
        bbox_x = self.from_delta(minpt, sz)
        return bbox_z, bbox_y, bbox_x

    @voxel_size.setter
    def voxel_size(self, vs: tuple):
        self._voxel_size = vs

    def slices_in_scale(self, voxel_size: tuple) -> tuple:
        """slices with specific voxel size volume

        Args:
            voxel_size (tuple): the target volume voxel size

        Returns:
            tuple: tuple of slices
        """
        minpt = tuple( p * s1 // s2 for p, s1, s2 in zip(
            self.minpt, self._voxel_size, voxel_size
        ))
        maxpt = tuple( p * s1 // s2 for p, s1, s2 in zip(
            self.maxpt, self._voxel_size, voxel_size
        ))
        bbox = Bbox(minpt, maxpt)
        return bbox.to_slices()


class BoundingBoxes(UserList):
    def __init__(self, bboxes: list) -> None:
        super().__init__()
        self.data = bboxes

    @classmethod
    def from_manual_setup(cls,
            chunk_size:Union[Vec, tuple], 
            chunk_overlap: Union[Vec, tuple]=Vec(0,0,0),
            roi_start: Union[Vec, tuple]=None, 
            roi_stop: Union[Vec, tuple]=None, 
            roi_size: Union[Vec, tuple]=None,
            grid_size: Union[Vec, tuple]=None,
            respect_chunk_size: bool = True,
            aligned_block_size: Union[Vec, tuple]=None,
            bounded: bool = False,
            layer_path: str = None,
            mip: int = 0):
        
        if layer_path:
            if layer_path.endswith('.h5'):
                assert os.path.exists(layer_path)
                with h5py.File(layer_path, mode='r') as file:
                    for key in file.keys():
                        if 'offset' in key:
                            roi_start = Vec(*(file[key]))
                        elif 'voxel_size' not in key:
                            if roi_size is None:
                                roi_size = Vec(*file[key].shape[-3:])
                if roi_start is None:
                    roi_start = Vec(0, 0, 0)
                roi_stop = roi_start + roi_size
            else:
                vol = CloudVolume(layer_path, mip=mip)
                # dataset shape as z,y,x
                dataset_size = vol.mip_shape(mip)[:3][::-1]
                dataset_offset = vol.mip_voxel_offset(mip)[::-1]
                if roi_size is None:
                    roi_size = Vec(*dataset_size)
                if roi_stop is None:
                    roi_stop = Vec(*[o+s for o, s in zip(dataset_offset, dataset_size)])
                if roi_start is None:
                    # note that we normally start from -overlap to keep the chunks aligned!
                    roi_start = dataset_offset - chunk_overlap
        assert roi_start is not None

        if roi_size is None and roi_stop is None and grid_size is None:
            grid_size = Vec(1, 1, 1)
        
        if isinstance(chunk_size, tuple):
            chunk_size = Vec(*chunk_size)
        if isinstance(chunk_overlap, tuple):
            chunk_overlap = Vec(*chunk_overlap)
        if isinstance(roi_start, tuple):
            roi_start = Vec(*roi_start)
        if isinstance(roi_size, tuple):
            roi_size = Vec(*roi_size)
        if isinstance(grid_size, tuple):
            grid_size = Vec(*grid_size)
        if isinstance(roi_stop, tuple):
            roi_stop = Vec(*roi_stop)
        
        stride = chunk_size - chunk_overlap
        if roi_stop is None:
            roi_stop = roi_start + stride*grid_size + chunk_overlap

        if aligned_block_size is not None:
            if not isinstance(aligned_block_size, Vec):
                aligned_block_size = Vec(*aligned_block_size)
            assert np.all(aligned_block_size <= chunk_size)
            assert np.alltrue(chunk_size % aligned_block_size == 0)
            roi_start -= roi_start % aligned_block_size
            assert len(aligned_block_size) == 3
            assert len(roi_stop) == 3
            for idx in range(3):
                if roi_stop[idx] % aligned_block_size[idx] > 0:
                    roi_stop[idx] += aligned_block_size[idx] - roi_stop[idx] % aligned_block_size[idx]

        if roi_size is None:
            roi_size = roi_stop - roi_start

        if grid_size is None:
            grid_size = (roi_size - chunk_overlap) / stride 
            grid_size = tuple(ceil(x) for x in grid_size)
            grid_size = Vec(*grid_size)

        # the stride should not be zero if there is more than one chunks
        for g, s in zip(grid_size, stride):
            if g > 1:
                assert s > 0
        
        final_output_stop = roi_start + (grid_size-1) * stride + chunk_size
        logging.info(f'\nroi start: {roi_start}')
        logging.info(f'stride: {stride}')
        logging.info(f'grid size: {grid_size}')
        logging.info(f'final output stop: {final_output_stop}')

        print(f'grid size: {grid_size} with {np.product(grid_size)} candidate bounding boxes.')

        bboxes = []
        for (gz, gy, gx) in itertools.product(
                    range(grid_size[0]), 
                    range(grid_size[1]),
                    range(grid_size[2])):
            chunk_start = roi_start + Vec(gz, gy, gx) * stride
            bbox = Bbox.from_delta(chunk_start, chunk_size)
            if not respect_chunk_size:
                bbox.maxpt = np.minimum(bbox.maxpt, roi_stop)
            if not bounded or np.all(tuple(m < p for m, p in zip(bbox.maxpt, roi_stop))):
                bboxes.append( bbox )

        print(f'get {len(bboxes)} bounding boxes as tasks.')
        return cls(bboxes)

    @classmethod
    def from_array(cls, arr: np.ndarray):
        bboxes = []
        for idx in range(arr.shape(0)):
            bbox = Bbox.from_vec(arr[idx, :])
            bboxes.append(bbox)

        return cls(bboxes)

    @classmethod
    def from_file(cls, file_path: str):
        arr = np.load(file_path)
        return cls.from_array(arr)

    def as_array(self) -> np.ndarray:
        task_num = len(self.data)

        arr = np.zeros((task_num, 6), dtype=np.int)
        for idx, bbox in enumerate(self.data):
            arr[idx, :3] = bbox.minpt
            arr[idx, 3:] = bbox.maxpt
        return arr

    def to_file(self, file_name: str) -> None:
        if file_name.endswith('.npy'):
            np.save(file_name, self.as_array())
        else:
            raise ValueError('only support npy format now.')
