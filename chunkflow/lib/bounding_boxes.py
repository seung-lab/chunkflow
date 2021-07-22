import logging
import os
from itertools import product
from collections import UserList
from math import ceil
from typing import Union
from copy import deepcopy

import numpy as np
import h5py 

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox


class BoundingBox(Bbox):
    def __init__(self, min_corner: list, max_corner: list, dtype=None, voxel_size: tuple = None):
        super().__init__(min_corner, max_corner, dtype=dtype)
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

    def clone(self):
        bbox = Bbox(self.minpt, self.maxpt, dtype=self.dtype)
        bbox = bbox.clone()
        return BoundingBox.from_bbox(bbox, voxel_size=self.voxel_size)

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

        print('grid size: ', grid_size)

        bboxes = []
        for (gz, gy, gx) in product(range(grid_size[0]), 
                                range(grid_size[1]),
                                range(grid_size[2])):
            chunk_start = roi_start + Vec(gz, gy, gx) * stride
            bbox = Bbox.from_delta(chunk_start, chunk_size)
            if not respect_chunk_size:
                bbox.maxpt = np.minimum(bbox.maxpt, roi_stop)
            bboxes.append( bbox )

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
