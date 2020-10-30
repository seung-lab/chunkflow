import os
from itertools import product
from collections import UserList

import numpy as np
import h5py 

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox


class BoundingBoxes(UserList):
    def __init__(self, bboxes: list) -> None:
        self.data = bboxes

    @classmethod
    def from_manual_setup(cls,
            chunk_size:tuple, chunk_overlap: tuple=(0,0,0),
            roi_start: tuple=None, roi_stop: tuple=None, layer_path: str=None,
            mip:int=0, grid_size: tuple=None, verbose: bool=True):
    
        if layer_path:
            if layer_path.endswith('.h5'):
                assert os.path.exists(layer_path)
                roi_size = None
                with h5py.File(layer_path, mode='r') as f:
                    for key in f.keys():
                        if 'offset' in key:
                            roi_start = Vec(*(f[key]))
                        else:
                            roi_size = Vec(*f[key].shape)
                if roi_start is None:
                    roi_start = Vec(0, 0, 0)
                roi_stop = roi_start + roi_size
            else:
                vol = CloudVolume(layer_path, mip=mip)
                # dataset shape as z,y,x
                dataset_size = vol.mip_shape(mip)[:3][::-1]
                dataset_offset = vol.mip_voxel_offset(mip)[::-1]
                if roi_stop is None:
                    roi_stop = Vec(*[o+s for o, s in zip(dataset_offset, dataset_size)])
                if roi_start is None:
                    # note that we normally start from -overlap to keep the chunks aligned!
                    roi_start = dataset_offset - chunk_overlap

        chunk_size = Vec(*chunk_size)
        chunk_overlap = Vec(*chunk_overlap)
        stride = chunk_size - chunk_overlap
        if isinstance(grid_size, tuple):
            grid_size = Vec(*grid_size)
    
        assert roi_start is not None
        if isinstance(roi_start, tuple):
            roi_start = Vec(*roi_start)
        
        if roi_stop is None:
            roi_stop = roi_start + stride*grid_size + chunk_overlap
        elif isinstance(roi_stop, tuple):
            roi_stop = Vec(*roi_stop)
        roi_size = roi_stop - roi_start

        if grid_size is None:
            grid_size = (roi_size - chunk_overlap) // stride + 1
        
        # the stride should not be zero if there is more than one chunks
        for g, s in zip(grid_size, stride):
            if g > 1:
                assert s > 0
        
        final_output_stop = roi_start + (grid_size-1) * stride + chunk_size
        if verbose:
            print('\nroi start: ', roi_start)
            print('stride: ', stride)
            print('grid size: ', grid_size)
            print('final output stop: ', final_output_stop)

        bboxes = []
        for (z, y, x) in product(range(grid_size[0]), 
                                range(grid_size[1]),
                                range(grid_size[2])):
            chunk_start = roi_start + Vec(z, y, x) * stride
            bbox = Bbox.from_delta(chunk_start, chunk_size)
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
