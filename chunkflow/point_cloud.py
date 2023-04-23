from __future__ import annotations
from functools import cached_property

import h5py
import numpy as np
from .lib.cartesian_coordinate import Cartesian, BoundingBox

class PointCloud:
    def __init__(self, points: np.ndarray, 
            voxel_size: Cartesian) -> None:
        assert points.ndim == 2
        assert points.shape[1] == 3

        if not isinstance(voxel_size, Cartesian):
            assert len(voxel_size) == 3
            voxel_size = Cartesian.from_collection(voxel_size)
        
        self.points = points
        self.voxel_size = voxel_size

    @classmethod
    def from_h5(cls, file_path: str) -> PointCloud:
        assert file_path.endswith('.h5')
        # assert h5py.is_hdf5(file_path)
        with h5py.File(file_path) as hf:
            points = np.asarray(hf['points'])
            voxel_size = Cartesian.from_collection(hf['voxel_size'])
        return cls(points, voxel_size)
    
    def to_h5(self, file_path: str):
        assert file_path.endswith('.h5')
        with h5py.File(file_path, 'w') as hf:
            hf['points'] = self.points
            hf['voxel_size'] = self.voxel_size

    @cached_property
    def bounding_box(self):
        bbox = BoundingBox.from_points(self.points)
        return bbox

    @cached_property
    def point_num(self):
        return self.points.shape[0] 