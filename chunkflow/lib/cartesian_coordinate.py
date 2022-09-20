# support the class method with parameter type of itself
from __future__ import annotations

import logging
import os
from collections import UserList, namedtuple
from math import ceil, floor
from typing import Union
from numbers import Number
import itertools
import re

from copy import deepcopy

import numpy as np
import h5py

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox

BOUNDING_BOX_RE = re.compile(r'(-?\d+)-(-?\d+)_(-?\d+)-(-?\d+)_(-?\d+)-(-?\d+)(?:\.gz|\.br|\.h5|\.json|\.npy|\.tif|\.csv|\.pkl|\.png|\.jpg)?$')

def to_cartesian(x: Union[tuple, list]):
    if x is None:
        return None
    else:
        assert len(x) == 3
        return Cartesian.from_collection(x)

class Cartesian(namedtuple('Cartesian', ['z', 'y', 'x'])):
    """Cartesian coordinate or offset."""
    __slots__ = ()

    @classmethod
    def from_collection(cls, col: Union[tuple, list, Vec]):
        return cls(*col)

    @property
    def ceil(self):
        return Cartesian(ceil(self.z), ceil(self.y), ceil(self.x))
    
    @property
    def floor(self):
        return Cartesian(floor(self.z), floor(self.y), floor(self.x))

    def __hash__(self):
        return hash((self.z, self.y, self.x))

    def __eq__(self, other: Union[int, tuple, Cartesian]) -> bool:
        if isinstance(other, int):
            return np.all([x==other for x in self])
        elif isinstance(other, Cartesian) or isinstance(other, tuple):
            return np.all([x==y for x, y in zip(self, other)])
        else:
            raise TypeError(f'only support int, tuple or Cartesian for now, but get {type(other)}')
    
    def __sub__(self, offset: Union[Cartesian, Number]):
        """subtract to another voxel coordinate

        Args:
            offset (Cartesian, int): another voxel coordinate
        """
        if isinstance(offset, Number):
            return Cartesian.from_collection([x-offset for x in self])
        else:
            return Cartesian.from_collection([x-o for x, o in zip(self, offset)])
    
    def __isub__(self, other:Union[int, Cartesian]):
        return self - other

    def __add__(self, offset: Union[Cartesian, tuple, int]):
        """add another coordinate

        Args:
            offset (Cartesian, int): offset
        """
        if isinstance(offset, int):
            return Cartesian(*[x+offset for x in self])
        else:
            return Cartesian(*[x+o for x, o in zip(self, offset)])
    
    def __iadd__(self, other: Union[Cartesian, int]):
        return self + other

    def __mul__(self, m: Union[Number, Cartesian]) -> Cartesian:
        if isinstance(m, Cartesian):
            return Cartesian.from_collection([x*y for x, y in zip(self, m)])
        elif isinstance(m, Number):
            return Cartesian(*[x*m for x in self])
        else:
            raise TypeError('only support number and Cartesian type.')

    def __imul__(self, other: Union[Cartesian, int]):
        return self * other

    def __floordiv__(self, d: Union[int, Cartesian]):
        if isinstance(d, Number):
            return Cartesian(*[x // d for x in self])
        else:
            return Cartesian.from_collection([x//d for x, d in zip(self, d)])

    def __ifloordiv__(self, other: Union[Cartesian, int]):
        return self // other

    def __truediv__(self, other: Union[Number, Cartesian]):
        if isinstance(other, Number):
            return Cartesian.from_collection([x/other for x in self])
        else:
            return Cartesian.from_collection([x/d for x, d in zip(self, other)])

    def __itruediv__(self, other: Union[Cartesian, int]):
        return self / other

    def __mod__(self, d: Union[int, Cartesian]) -> Cartesian:
        if isinstance(d, int):
            return Cartesian(*[x%d for x in self])
        elif isinstance(d, Cartesian):
            return Cartesian.from_collection([x%y for x, y in zip(self, d)])
        else:
            raise TypeError('only support int or Cartesian for now.')
    
    def __imod__(self, other: Union[Cartesian, int]):
        return self % other

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

    def __neg__(self) -> Cartesian:
        return Cartesian(-self.z, -self.y, -self.x)

    #def __setitem__(self, key: int, value: int):
    #    if key == 0:
    #        self.z = value
    #    elif key == 1:
    #        self.y = value
    #    elif key == 2:
    #        self.z = value
    #    else:
    #        raise KeyError('invalide key. only support 0,1,2.')

    @property
    def vec(self):
        return Vec(*self)

    @property
    def tuple(self):
        return (self.z, self.y, self.x)


class BoundingBox(Bbox):
    def __init__(self, 
            minpt: Union[list, Cartesian],
            maxpt: Union[list, Cartesian],
            dtype: np.dtype =None):
        if isinstance(minpt, Cartesian):
            minpt = minpt.vec
        
        if isinstance(maxpt, Cartesian):
            maxpt = maxpt.vec
        super().__init__(minpt, maxpt, dtype=dtype)

    @classmethod
    def from_bbox(cls, bbox: Bbox):
        return cls(bbox.minpt, bbox.maxpt)
    
    @classmethod
    def from_string(cls, string: str):
        match = BOUNDING_BOX_RE.search(string)
        if match is None:
            return None
        else:
            zstart, zstop, ystart, ystop, xstart, xstop = map(
                int, match.groups()
            )
            start = Cartesian(zstart, ystart, xstart)
            stop = Cartesian(zstop, ystop, xstop)
            return cls(start, stop)

    @classmethod
    def from_delta(cls, 
            minpt: Union[list, tuple, Cartesian, np.ndarray], 
            plus: Union[list, tuple, Cartesian, np.ndarray]):
        bbox = super().from_delta(minpt, plus)
        return cls.from_bbox(bbox)

    @classmethod
    def from_list(cls, lst: list):
        bbox = Bbox.from_list(lst)
        return cls.from_bbox(bbox)

    @classmethod
    def from_vec(cls, arr: np.ndarray):
        assert len(arr) == 3
        bbox = Bbox.from_vec(arr)
        return cls.from_bbox(bbox)

    @classmethod
    def from_points(cls, arr: np.ndarray):
        bbox = Bbox.from_points(arr)
        return cls.from_bbox(bbox)

    @classmethod
    def from_slices(cls, tpl: tuple):
        bbox = Bbox.from_slices(tpl)
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

    @property
    def string(self):
        return self.to_filename()

    @property
    def start(self):
        return Cartesian.from_collection(self.minpt)
    
    @property
    def stop(self):
        return Cartesian.from_collection(self.maxpt)

    @property
    def shape(self):
        return self.stop - self.start
        
    def __repr__(self):
        return f'BoundingBox({self.minpt}, {self.maxpt}, dtype={self.dtype}'

    def clone(self):
        bbox = Bbox(self.minpt, self.maxpt, dtype=self.dtype)
        bbox = bbox.clone()
        return BoundingBox.from_bbox(bbox)

    def adjust(self, size: Union[Cartesian, int, tuple, list, Vec]):
        if size is None:
            logging.warn('adjusting bounding box size is None!')
            return self

        if not isinstance(size, int):
            assert len(size)==3 or len(size)==6
            size = Vec(*size)
        else:
            size = Cartesian(size, size, size)
        self.minpt -= size[:3]
        self.maxpt += size[-3:]
        return self

    def adjust_corner(self, corner_offset: Union[tuple, list]):
        if corner_offset is not None:
            assert len(corner_offset) == 6
            self.minpt += Cartesian.from_collection(corner_offset[:3])
            self.maxpt += Cartesian.from_collection(corner_offset[-3:])

    def union(self, bbox2):
        """Merge another bounding box

        Args:
            bbox2 (Union[BoundingBox, Bbox]): another bounding box

        Returns:
            BoundingBox: bounding box after merging
        """

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

    
class BoundingBoxes(UserList):
    def __init__(self, bboxes: list) -> None:
        super().__init__()
        self.data = bboxes

    @classmethod
    def from_manual_setup(cls,
            chunk_size:Union[Vec, tuple], 
            chunk_overlap: Union[Vec, tuple, Cartesian]=Cartesian(0,0,0),
            roi_start: Union[Vec, tuple, Cartesian]=None, 
            roi_stop: Union[Vec, tuple, Cartesian]=None, 
            roi_size: Union[Vec, tuple, Cartesian]=None,
            grid_size: Union[Vec, tuple, Cartesian]=None,
            respect_chunk_size: bool = True,
            aligned_block_size: Union[Vec, tuple, Cartesian]=None,
            bounded: bool = False,
            layer_path: str = None,
            mip: int = 0):

        if not layer_path:
            if grid_size is None and roi_size is None and roi_stop is None:
                grid_size = Cartesian(1, 1, 1)

            if roi_start is None:
                roi_start = Cartesian(0, 0, 0)
            elif not isinstance(roi_start, Cartesian):
                roi_start = Cartesian.from_collection(roi_start)
            if roi_size is None and chunk_size is not None:
                roi_size = Cartesian.from_collection(chunk_size)
            roi_stop = roi_start + roi_size
        else:
            if layer_path.endswith('.h5'):
                assert os.path.exists(layer_path)
                with h5py.File(layer_path, mode='r') as file:
                    for key in file.keys():
                        if 'offset' in key:
                            roi_start = Cartesian(*(file[key]))
                        elif 'voxel_size' not in key:
                            if roi_size is None:
                                roi_size = Cartesian(*file[key].shape[-3:])
                if roi_start is None:
                    roi_start = Cartesian(0, 0, 0)
                if roi_size is None and chunk_size is not None:
                    roi_size = Cartesian.from_collection(chunk_size)

                roi_stop = roi_start + roi_size
            else:
                vol = CloudVolume(layer_path, mip=mip)
                # dataset shape as z,y,x
                dataset_size = vol.mip_shape(mip)[:3][::-1]
                dataset_offset = vol.mip_voxel_offset(mip)[::-1]
                dataset_size = Cartesian.from_collection(dataset_size)
                dataset_offset = Cartesian.from_collection(dataset_offset)

                if roi_size is None:
                    roi_size = dataset_size
                if roi_stop is None:
                    roi_stop = dataset_offset + dataset_size
                if roi_start is None:
                    # note that we normally start from -overlap to keep the chunks aligned!
                    roi_start = dataset_offset - chunk_overlap
        assert roi_start is not None

        if not isinstance(chunk_size, Cartesian):
            chunk_size = Cartesian(*chunk_size)
        if not isinstance(chunk_overlap, Cartesian):
            chunk_overlap = Cartesian(*chunk_overlap)
        if not isinstance(roi_start, Cartesian):
            assert len(roi_start) == 3
            roi_start = Cartesian(*roi_start)
        if not isinstance(roi_size, Cartesian):
            roi_size = Cartesian(*roi_size)
        if grid_size is not None and not isinstance(grid_size, Cartesian):
            grid_size = Cartesian(*grid_size)
        if not isinstance(roi_stop, Cartesian):
            roi_stop = Cartesian(*roi_stop)

        stride = chunk_size - chunk_overlap
        if roi_stop is None:
            roi_stop = roi_start + stride*grid_size + chunk_overlap

        if aligned_block_size is not None:
            if not isinstance(aligned_block_size, Vec):
                aligned_block_size = Cartesian(*aligned_block_size)
            assert np.all(aligned_block_size <= chunk_size)
            assert chunk_size % aligned_block_size == 0
            roi_start -= roi_start % aligned_block_size
            assert len(aligned_block_size) == 3
            assert len(roi_stop) == 3
            roi_stop_temp = [x for x in roi_stop]
            for idx in range(3):
                if roi_stop[idx] % aligned_block_size[idx] > 0:
                    roi_stop_temp[idx] += aligned_block_size[idx] - roi_stop[idx] % aligned_block_size[idx]
            roi_stop = Cartesian.from_collection(roi_stop_temp)

        if roi_size is None:
            roi_size = roi_stop - roi_start

        if grid_size is None:
            grid_size = (roi_size - chunk_overlap) / stride 
            grid_size = grid_size.ceil
            # grid_size = Cartesian.from_collection([ceil(x) for x in grid_size])

        # the stride should not be zero if there is more than one chunks
        for g, s in zip(grid_size, stride):
            if g > 1:
                assert s > 0

        final_output_stop = roi_start + (grid_size-1) * stride + chunk_size
        logging.info(f'\nroi start: {roi_start}')
        logging.info(f'stride: {stride}')
        logging.info(f'grid size: {grid_size}')
        logging.info(f'final output stop: {final_output_stop}')
        center_index = grid_size.z // 2 * (grid_size.y*grid_size.x) + \
            grid_size.y // 2 * grid_size.x + \
                grid_size.x // 2
        logging.info(f'center chunk index: {center_index}')
        print(f'center chunk index: {center_index}')

        logging.info(f'grid size: {grid_size} with {np.product(grid_size)} candidate bounding boxes.')

        bboxes = []
        for (gz, gy, gx) in itertools.product(
                    range(grid_size[0]), 
                    range(grid_size[1]),
                    range(grid_size[2])):
            chunk_start = roi_start + Cartesian(gz, gy, gx) * stride
            bbox = BoundingBox.from_delta(chunk_start, chunk_size)
            if not respect_chunk_size:
                bbox.maxpt = np.minimum(bbox.maxpt, Vec(*roi_stop))
            if not bounded or np.all(tuple(m < p for m, p in zip(bbox.maxpt, roi_stop))):
                bboxes.append( bbox )

        logging.info(f'get {len(bboxes)} bounding boxes as tasks.')
        return cls(bboxes)

    @classmethod
    def from_array(cls, arr: np.ndarray, is_zyx: bool=True):
        if not is_zyx:
            # the coordinate is xyz,xyz
            arr[:, :] = arr[:, [2,1,0,5,4,3]]

        bboxes = []
        for idx in range(arr.shape[0]):
            bbox = BoundingBox.from_list(arr[idx, :])
            bboxes.append(bbox)

        return cls(bboxes)

    @classmethod
    def from_file(cls, file_path: str, is_zyx: bool=True):

        if file_path.endswith('.npy'):
            arr = np.load(file_path)
        elif file_path.endswith('.txt'):
            arr = np.loadtxt(file_path, dtype=int, delimiter=',')
        return cls.from_array(arr, is_zyx=is_zyx)

    def as_array(self) -> np.ndarray:
        task_num = len(self.data)

        arr = np.zeros((task_num, 6), dtype=int)
        for idx, bbox in enumerate(self.data):
            arr[idx, :3] = bbox.minpt
            arr[idx, 3:] = bbox.maxpt
        return arr

    def to_file(self, file_name: str) -> None:
        if file_name.endswith('.npy'):
            np.save(file_name, self.as_array())
        else:
            raise ValueError('only support npy format now.')

    def __len__(self):
        return len(self.data)