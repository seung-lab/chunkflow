from __future__ import annotations

from typing import Union

import numpy as np

from .cartesian_coordinate import Cartesian, BoundingBox


class RegionOfInterest(BoundingBox):
    """Physical Region of Interest. 
    Basically Bounding Box with voxel size.

    Args:
        BoundingBox (BoundingBox): _description_
    """
    def __init__(self, 
            minpt: Union[list, Cartesian], maxpt: Union[list, Cartesian],
            voxel_size: Cartesian, 
            dtype=None):
        """Physical region of interest.
        Basically BoundingBox with voxel size.

        Args:
            minpt (Union[list, Cartesian]): the start of bounding box
            maxpt (Union[list, Cartesian]): the stop of bounding box
            voxel_size (Cartesian): the physical size of each voxel/unit. We normally use nanometer here.
            dtype (_type_, optional): integer datatype of bounding box. Defaults to None.
        """
        super().__init__(minpt, maxpt, dtype)

        self.voxel_size = voxel_size

    @classmethod
    def from_bbox(cls, bbox: BoundingBox, voxel_size: Cartesian):
        return cls(bbox.minpt, bbox.maxpt, voxel_size)

    def __repr__(self):
        return f'from {self.minpt} to {self.maxpt}, dtype={self.dtype}, voxel_size={self.voxel_size})'

    @property
    def bounding_box(self):
        return BoundingBox(self.minpt, self.maxpt, dtype=self.dtype)

    @property
    def physical_size(self):
        return self.voxel_size * self.shape 

    def clone(self):
        bbox = self.bounding_box.clone()
        # the Cartesian is immutable, so we do not need to clone it.
        return RegionOfInterest(bbox.minpt, bbox.maxpt, self.voxel_size, dtype=bbox.dtype)

    def slices_in_scale(self, voxel_size: Cartesian) -> tuple:
        """slices with specific voxel size volume

        Args:
            voxel_size (Cartesian): the target volume voxel size

        Returns:
            tuple: tuple of slices
        """
        minpt = tuple( p * s1 // s2 for p, s1, s2 in zip(
            self.minpt, self.voxel_size, voxel_size
        ))
        maxpt = tuple( p * s1 // s2 for p, s1, s2 in zip(
            self.maxpt, self.voxel_size, voxel_size
        ))
        bbox = BoundingBox(minpt, maxpt)
        return bbox.slices


class ROITree:
    def __init__(self, roi: RegionOfInterest, axis: int,
            left: ROITree, right: ROITree):
        """Split the volume hierarchically using a modified aligned K-D tree

        Args:
            roi (RegionOfInterest): physical region covered
            axis (int): the splitting dimension of left and right roi
            left (ROIKDTree): left roi with lower coordinate values in axis dim
            right (ROIKDTree): right roi with higher coordinate values in axis dim
        """
        assert axis >= 0 and axis < 3 
        self.roi = roi
        self.axis = axis
        self.left = left
        self.right = right

    @classmethod
    def from_roi(cls, roi: RegionOfInterest, factor: Cartesian,
            atomic_block_size: Cartesian, atomic_voxel_size: Cartesian):
        """Generate the ROITree from a single roi.
        This roi is not required to be aligned with the atomic block size.
        If it is not aligned, a roi will partially cover the volume.

        Args:
            roi (RegionOfInterest): the total roi covered by this tree.
            factor (Cartesian): downsampling factor in each level.
            atomic_block_size (Cartesian): the size of the leaf node/block
            atomic_voxel_size (Cartesian): the voxel size of leaf block
        """
        pass

        # assert roi.voxel_size % atomic_voxel_size == Cartesian(0, 0, 0)
        # assert roi.voxel_size // atomic_voxel_size % factor == Cartesian(0, 0, 0)
        
        # if roi.voxel_size == atomic_voxel_size:
        #     # this is the leaf roi/block
        #     return cls(roi, None, None, None)

        # # find the relatively longest axis to split
        # children_voxel_size = roi.voxel_size // factor
        # block_nums = roi.physical_size / (children_voxel_size *  )
        # block_nums = np.ceil(block_nums)
        # axis = np.argmax(block_nums)

        # # split along axis
        # left_start = roi.start * factor
        # left_block_nums = 
        # left_stop = left_start + 
        # left_roi = RegionOfInterest()
        # left = cls.from_roi(left_roi, factor, atomic_block_size, atomic_voxel_size)



    @property
    def is_leaf(self):
        return self.axis is None        