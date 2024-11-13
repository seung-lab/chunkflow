# support the class method with parameter type of itself
from __future__ import annotations

import random

import os
from collections import UserList, namedtuple
from math import ceil, floor
from typing import Union, List
from numbers import Number
import itertools
from functools import cached_property
import re
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import h5py

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox

BOUNDING_BOX_RE = re.compile(r'(-?\d+)-(-?\d+)_(-?\d+)-(-?\d+)_(-?\d+)-(-?\d+)(?:\.gz|\.br|\.h5|\.json|\.npy|\.tif|\.csv|\.pkl|\.png|\.jpg)?$')


def to_cartesian(x: Union[tuple, list, None]):
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
        assert len(col) == 3
        return cls(*col)

    @property
    def ceil(self):
        return Cartesian(ceil(self.z), ceil(self.y), ceil(self.x))
    
    @property
    def floor(self):
        return Cartesian(floor(self.z), floor(self.y), floor(self.x))
    
    def __hash__(self):
        return hash((self.z, self.y, self.x))

    def __eq__(self, other: Union[Number, tuple, Cartesian]) -> bool:
        if isinstance(other, Number):
            return np.all([x==other for x in self])
        elif isinstance(other, Cartesian) or isinstance(other, tuple):
            return np.all([x==y for x, y in zip(self, other)])
        else:
            raise TypeError(f'only support Number , tuple or Cartesian for now, but get {type(other)}')
    
    def __sub__(self, offset: Union[Cartesian, Number]):
        """subtract to another voxel coordinate

        Args:
            offset (Cartesian, Number): another voxel coordinate
        """
        if isinstance(offset, Number):
            return Cartesian.from_collection([x-offset for x in self])
        else:
            return Cartesian.from_collection([x-o for x, o in zip(self, offset)])
    
    def __isub__(self, other:Union[Number, Cartesian]):
        return self - other

    def __add__(self, offset: Union[Cartesian, tuple, Number ]):
        """add another coordinate

        Args:
            offset (Cartesian, Number): offset
        """
        if isinstance(offset, Number):
            return Cartesian(*[x+offset for x in self])
        else:
            return Cartesian(*[x+o for x, o in zip(self, offset)])
    
    def __iadd__(self, other: Union[Cartesian, Number]):
        return self + other

    def __mul__(self, m: Union[Number, Cartesian]) -> Cartesian:
        if isinstance(m, Cartesian):
            return Cartesian.from_collection([x*y for x, y in zip(self, m)])
        elif isinstance(m, Number):
            return Cartesian(*[x*m for x in self])
        else:
            raise TypeError('only support number and Cartesian type.')

    def __imul__(self, other: Union[Cartesian, Number]):
        return self * other

    def __floordiv__(self, d: Union[Number, Cartesian]):
        if isinstance(d, Number):
            return Cartesian(*[x // d for x in self])
        else:
            return Cartesian.from_collection([x//d for x, d in zip(self, d)])

    def __ifloordiv__(self, other: Union[Cartesian, Number]):
        return self // other

    def __truediv__(self, other: Union[Number, Cartesian]):
        if isinstance(other, Number):
            return Cartesian.from_collection([x/other for x in self])
        else:
            return Cartesian.from_collection([x/d for x, d in zip(self, other)])

    def __itruediv__(self, other: Union[Cartesian, Number]):
        return self / other

    def __mod__(self, d: Union[Number, Cartesian]) -> Cartesian:
        if isinstance(d, Number):
            return Cartesian(*[x%d for x in self])
        elif isinstance(d, Cartesian):
            return Cartesian.from_collection([x%y for x, y in zip(self, d)])
        else:
            raise TypeError(f'only support Number or Cartesian for now, but got {type(d)}')
    
    def __imod__(self, other: Union[Cartesian, Number]):
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
    
    @property
    def list(self):
        return [self.z, self.y, self.x]

    @property
    def inverse(self):
        return Cartesian(self.x, self.y, self.z)


@dataclass(frozen=True)
class BoundingBox:
    start: Cartesian
    stop: Cartesian
    # def __post_init__(self, start, stop) -> BoundingBox:
    #     if not isinstance(start, Cartesian):
    #         assert len(start) == 3
    #         start = Cartesian.from_collection(start)
    #     if not isinstance(stop, Cartesian):
    #         assert len(stop) == 3
    #         stop = Cartesian.from_collection(stop)
    #     self.__setattr__('start', start)
    #     self.__setattr__('stop', stop)
        
    @classmethod
    def from_bbox(cls, bbox: Bbox):
        assert isinstance(bbox, Bbox)
        start = Cartesian.from_collection(bbox.minpt)
        stop  = Cartesian.from_collection(bbox.maxpt)
        return cls(start, stop)
    
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
            start: Union[list, tuple, Cartesian, np.ndarray], 
            plus: Union[list, tuple, Cartesian, np.ndarray]):
        if not isinstance(start, Cartesian):
            start = Cartesian.from_collection(start)
        if not isinstance(plus, Cartesian):
            plus = Cartesian.from_collection(plus)
        stop = start + plus
        return cls(start, stop)

    @classmethod
    def from_list(cls, lst: list):
        start = Cartesian.from_collection(lst[:3])
        stop  = Cartesian.from_collection(lst[-3:])
        return cls(start, stop)

    @classmethod
    def from_points(cls, arr: np.ndarray):
        bbox = Bbox.from_points(arr)
        return cls.from_bbox(bbox)

    @classmethod
    def from_slices(cls, tpl: tuple):
        bbox = Bbox.from_slices(tpl)
        return cls.from_bbox(bbox)

    @classmethod
    def from_center(cls, center: Cartesian, extent: Number,
            even_size: bool = True):
        """Create bounding box from center and extent

        Args:
            center (Cartesian): center coordinate
            extent (Number): the range to extent, like radius
            even_size (bool): produce even size or odd size including the center.
        """
        start = center - extent
        # the maxpt is not inclusive, so we need +1
        stop = center + extent
        if not even_size:
            # this will make the size to be odd
            stop += 1
        return cls(start, stop)

    @cached_property
    def dvid_roi(self):
        return [self.start.tuple, self.stop.tuple]

    @cached_property
    def string(self):
        bbox = Bbox(self.start, self.stop)
        return bbox.to_filename()
    
    # @property
    # def center(self) -> Cartesian:
    #     ct = (self.minpt + self.maxpt) // 2
    #     ct = Cartesian.from_collection(ct)
    #     return ct

    @property
    def minpt(self) -> Cartesian:
        return self.start
    
    @property
    def maxpt(self) -> Cartesian:
        return self.stop

    @cached_property
    def slices(self):
        return (
            slice(self.start.z, self.stop.z),
            slice(self.start.y, self.stop.y),
            slice(self.start.x, self.stop.x),
        )

    @property
    def random_coordinate(self) -> Cartesian:
        """find a random coordinate inside this bounding box

        Returns:
            Cartesian: the coordinate inside this bounding box
        """
        z = random.randrange(self.start.z, self.stop.z)
        y = random.randrange(self.start.y, self.stop.y)
        x = random.randrange(self.start.x, self.stop.x)
        return Cartesian(z, y, x)

    @cached_property
    def shape(self):
        return self.stop - self.start

    def decompose_to_aligned_block_bounding_boxes(self, 
            block_size: Cartesian, bounded: bool=True) -> BoundingBoxes:
        bboxes = BoundingBoxes()
        if bounded:
            for z in range(self.start.z, self.stop.z, block_size.z):
                for y in range(self.start.y, self.stop.y, block_size.y):
                    for x in range(self.start.x, self.stop.x, block_size.x):
                        bbox = BoundingBox.from_delta(Cartesian(z,y,x), block_size)
                        bboxes.append(bbox)
        else:
            for z in range(self.start.z, self.stop.z+block_size.z-1, block_size.z):
                for y in range(self.start.y, self.stop.y+block_size.y-1, block_size.y):
                    for x in range(self.start.x, self.stop.x+block_size.x-1, block_size.x):
                        bbox = BoundingBox.from_delta(Cartesian(z,y,x), block_size)
                        bboxes.append(bbox)
        return bboxes
    
    def decompose_to_unaligned_block_bounding_boxes(self, 
            block_size: Cartesian) -> BoundingBoxes:
        bboxes = BoundingBoxes()
        for z in range(self.start.z, self.stop.z, block_size.z):
            for y in range(self.start.y, self.stop.y, block_size.y):
                for x in range(self.start.x, self.stop.x, block_size.x):
                    block_start = Cartesian(z,y,x)
                    block_stop = block_start + block_size
                    # the block stop can not exceed the boundary of volume
                    block_stop = Cartesian.from_collection(
                        np.minimum(block_stop, self.stop)
                    )
                    bbox = BoundingBox(block_start, block_stop)
                    bboxes.append(bbox)
        return bboxes
       
    def __repr__(self):
        return f'BoundingBox({self.start}, {self.stop}'

    def __mul__(self, operand: Cartesian | Number):
        assert isinstance(operand, Number) or isinstance(operand, Cartesian)
        start = self.start * operand
        stop = self.stop * operand
        return BoundingBox(start, stop)

    def __floordiv__(self, other: Number | Cartesian | BoundingBox) -> BoundingBox:
        if isinstance(other, Number) or isinstance(other, Cartesian):
            minpt = self.minpt // other
            maxpt = self.maxpt // other
        elif isinstance(other, BoundingBox):
            minpt = self.minpt // other.minpt
            maxpt = self.maxpt // other.maxpt
        elif isinstance(other, np.ndarray):
            other = Cartesian.from_collection(other)
            minpt = self.start // other
            maxpt = self.stop // other
        else:
            raise ValueError(f'unsupported type of operand: {type(other)}')
        return BoundingBox(minpt, maxpt)
         
    def __ifloordiv__(self, other: BoundingBox | Number | Cartesian):
        return self // other

    def inverse_order(self):
        return BoundingBox(
            self.start.inverse, 
            self.stop.inverse
        )

    def __add__(self, other: Cartesian | Number):
        start = self.start + other
        stop = self.stop + other
        return BoundingBox(start, stop)

    def __iadd__(self, other: Cartesian | Number):
        start = self.start + other
        stop = self.stop + other
        return BoundingBox(start, stop)

    def clone(self):
        return BoundingBox(self.start, self.stop)

    def adjust(self, size: Union[Cartesian, Number, tuple, list, Vec]):
        """adjust the range of bounding box

        Args:
            size (Union[Cartesian, Number, tuple, list, Vec]): adjust size

        Returns:
            BoundingBox: adjusted bounding box
        """
        if size is None:
            print('adjusting bounding box size is None!')
            return self

        if isinstance(size, Number):
            size = (-size, -size, -size, size, size, size)
        elif len(size) == 3:
            size = (-size[0], -size[1], -size[2], size[0], size[1], size[2])
        else:
            assert len(size) == 6

        start = self.minpt + size[:3]
        stop  = self.maxpt + size[-3:]
        return BoundingBox(start, stop)

    def adjust_corner(self, corner_offset: Union[tuple, list]):
        assert corner_offset is not None
        assert len(corner_offset) == 6
        start = self.start + Cartesian.from_collection(corner_offset[:3])
        stop  = self.stop + Cartesian.from_collection(corner_offset[-3:])
        return BoundingBox(start, stop)

    def union(self, bbox2: BoundingBox):
        """Merge another bounding box

        Args:
            bbox2 (Union[BoundingBox, Bbox]): another bounding box

        Returns:
            BoundingBox: bounding box after merging
        """
        minpt = np.minimum(self.minpt, bbox2.minpt)
        maxpt = np.maximum(self.maxpt, bbox2.maxpt)
        start = Cartesian.from_collection(minpt)
        stop  = Cartesian.from_collection(maxpt)
        return BoundingBox(start, stop)

    def intersection(self, bbox2: BoundingBox):
        minpt = np.maximum(self.minpt, bbox2.minpt)
        maxpt = np.minimum(self.maxpt, bbox2.maxpt)
        start = Cartesian.from_collection(minpt)
        stop  = Cartesian.from_collection(maxpt)
        return BoundingBox(start, stop)

    def contains(self, point: Union[tuple, Vec, list]):
        assert 3 == len(point)
        return np.all(np.asarray(
            (self.maxpt >= Vec(*point)))) and np.all(
                np.asarray((self.minpt <= Vec(*point)))) 

    def decompose(self, block_size: Cartesian, 
            ignore_unaligned: bool = True) -> BoundingBoxes:
        """decompose the bounding box to a list of bounding boxes
        If there exist some space that can not fit in a whole block, it will be 
        ignored! 

        Args:
            block_size (Cartesian): the decomposed block size.

        Returns:
            BoundingBoxes: a list of bounding boxes
        """
        assert ignore_unaligned 
        if not isinstance(block_size, Cartesian):
            block_size = Cartesian.from_collection(block_size)
        
        bboxes = BoundingBoxes()

        for z in range(self.start.z, self.stop.z-block_size.z+1, block_size.z):
            for y in range(self.start.y, self.stop.y-block_size.y+1, block_size.y):
                for x in range(self.start.x, self.stop.x-block_size.x+1, block_size.x):
                    bbox = BoundingBox.from_delta(Cartesian(z,y,x), block_size)
                    bboxes.append(bbox)
        return bboxes
    
    @cached_property
    def slices(self):
        return tuple(slice(x0, x1) for x0, x1 in zip(self.start, self.stop))

    @cached_property
    def cloud_volume_bbox_xyz(self):
        return Bbox.from_slices(self.slices[::-1])

    @cached_property
    def shape(self):
        return self.stop - self.start

    @cached_property
    def left_neighbors(self):
        sz = self.shape

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
    
    def is_aligned_with(self, block_shape: Union[tuple, Cartesian]) -> bool:
        """whether the bounding box is aligned with block size or not

        Args:
            block_shape (Union[tuple, Cartesian]): size or shape of a volume block

        Returns:
            is_aligned (bool): whether it is aligned or not
        """
        is_start_aligned = (self.start % block_shape == Cartesian(0,0,0))
        is_stop_aligned = (self.stop % block_shape == Cartesian(0,0,0))
        return is_start_aligned and is_stop_aligned

    
class BoundingBoxes(UserList):
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
            volume_path: str = None,
            mip: int = 0,
            use_https: bool = False,
            ):

        if not volume_path:
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
            if volume_path.endswith('.h5'):
                assert os.path.exists(volume_path)
                with h5py.File(volume_path, mode='r') as file:
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
                vol = CloudVolume(volume_path, mip=mip, use_https=use_https)
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
        assert chunk_size is not None
        
        if not isinstance(chunk_size, Cartesian):
            chunk_size = Cartesian.from_collection(chunk_size)
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
        print(f'\nroi start: {roi_start}')
        print(f'stride: {stride}')
        print(f'grid size: {grid_size}')
        print(f'final output stop: {final_output_stop}')
        center_index = grid_size.z // 2 * (grid_size.y*grid_size.x) + \
            grid_size.y // 2 * grid_size.x + \
                grid_size.x // 2

        print(f'grid size: {grid_size} with {np.product(grid_size)} candidate bounding boxes.')

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

        print(f'get {len(bboxes)} bounding boxes as tasks.')
        print(f'center chunk index: {center_index} with bounding box: {bboxes[center_index].string}')
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

    @property
    def array(self) -> np.ndarray:
        task_num = len(self.data)

        arr = np.zeros((task_num, 6), dtype=int)
        for idx, bbox in enumerate(self.data):
            arr[idx, :3] = bbox.minpt
            arr[idx, 3:] = bbox.maxpt
        return arr

    def to_file(self, file_name: str) -> None:
        if file_name.endswith('.npy'):
            np.save(file_name, self.array)
        else:
            raise ValueError('only support npy format now.')

    def __len__(self):
        return len(self.data)


@dataclass(frozen=True)
class PhysicalBoundingBox(BoundingBox):
    voxel_size: Cartesian
   
    @classmethod
    def from_bounding_box(cls, bbox: BoundingBox, 
            voxel_size: Cartesian) -> PhysicalBoundingBox:
        return cls(bbox.start, bbox.stop, 
            voxel_size)
        
    @cached_property
    def voxel_bounding_box(self) -> BoundingBox:
        return BoundingBox(self.start, self.stop)
    
    def to_other_voxel_size(self, voxel_size2: Cartesian) -> PhysicalBoundingBox:
        assert voxel_size2 != self.voxel_size
        
        if voxel_size2 >= self.voxel_size:

            factors = voxel_size2 // self.voxel_size
            start = self.start // factors
            stop = self.stop // factors
        else:
            factors = self.voxel_size // voxel_size2
            start = self.start * factors
            stop = self.stop * factors
        return PhysicalBoundingBox(start, stop, voxel_size2)
