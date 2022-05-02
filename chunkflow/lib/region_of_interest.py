from __future__ import annotations

from math import ceil
from typing import Union

import numpy as np

from .bounding_boxes import Cartesian, BoundingBox


class RegionOfInterest(BoundingBox):
    """Physical Region of Interest. 
    Basically Bounding Box with voxel size.

    Args:
        BoundingBox (BoundingBox): _description_
    """
    def __init__(self, 
            voxel_start: Union[list, Cartesian, tuple],
            voxel_stop: Union[list, Cartesian, tuple],
            voxel_size: Cartesian):
        """Physical region of interest.
        Basically BoundingBox with voxel size.

        Args:
            voxel_start (Union[list, Cartesian, tuple]): the start of bounding box
            voxel_start (Union[list, Cartesian, tuple]): the stop of bounding box
            voxel_size (Cartesian): the physical size of each voxel/unit. We normally use nanometer here.
        """
        super().__init__(voxel_start, voxel_stop)

        self.voxel_size = voxel_size

    @classmethod
    def from_bounding_box(cls, bbox: BoundingBox, voxel_size: Cartesian):
        return cls(bbox.start, bbox.stop, voxel_size)

    def __repr__(self):
        return f'from {self.minpt} to {self.maxpt}, dtype={self.dtype}, voxel_size={self.voxel_size})'

    @property
    def bounding_box(self):
        return BoundingBox(self.start, self.stop, dtype=self.dtype)

    @property
    def voxel_start(self):
        return self.start

    @property
    def voxel_stop(self):
        return self.stop

    @property
    def voxel_shape(self):
        return self.voxel_stop - self.voxel_start

    @property
    def physical_size(self):
        return self.voxel_size * self.shape 

    def clone(self):
        bbox = self.bounding_box.clone()
        # the Cartesian is immutable, so we do not need to clone it.
        return RegionOfInterest(bbox.start, bbox.stop, self.voxel_size)

    def slices_in_scale(self, voxel_size: Cartesian) -> tuple:
        """slices with specific voxel size volume

        Args:
            voxel_size (Cartesian): the target volume voxel size

        Returns:
            tuple: tuple of slices
        """
        minpt = tuple( p * s1 // s2 for p, s1, s2 in zip(
            self.voxel_start, self.voxel_size, voxel_size
        ))
        maxpt = tuple( p * s1 // s2 for p, s1, s2 in zip(
            self.voxel_stop, self.voxel_size, voxel_size
        ))
        bbox = BoundingBox(minpt, maxpt)
        return bbox.to_slices()


class ROITree:
    """The binary tree of region of interest. This is used to distribute tasks with dependency.
    """
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
    def from_atomic_roi(
            cls, roi: RegionOfInterest, factor: Cartesian,
            leaf_block_size: Cartesian):
        raise NotImplementedError()

    @classmethod
    def from_root_roi(
        cls, roi: RegionOfInterest, factor: Cartesian,
        leaf_block_size: Cartesian, leaf_voxel_size: Cartesian):
        """Generate the ROITree from a single roi.
        This roi is not required to be aligned with the leaf block size.
        If it is not aligned, a roi will partially cover the volume.

        Args:
            roi (RegionOfInterest): the total roi covered by this tree.
            factor (Cartesian): downsampling factor in each level.
            leaf_block_size (Cartesian): the size of the leaf node/block
            leaf_voxel_size (Cartesian): the voxel size of leaf block
        """
        assert roi.voxel_size % leaf_voxel_size == Cartesian(0, 0, 0)
        assert roi.voxel_size // leaf_voxel_size % factor == Cartesian(0, 0, 0)
        
        if roi.voxel_size == leaf_voxel_size:
            # this is the leaf roi/block
            return cls(roi, None, None, None)

        # find the relatively longest axis to split
        leaf_physical_size = leaf_voxel_size * leaf_block_size
        block_nums = roi.physical_size / leaf_physical_size
        block_nums = np.ceil(block_nums)
        axis = np.argmax(block_nums)

        children_voxel_size = roi.voxel_size // factor
        children_voxel_start = roi.voxel_start * factor 
        children_volume_shape = roi.voxel_shape * factor
        children_voxel_stop = children_voxel_start + \
            children_volume_shape
        
        # split along axis
        left_voxel_start = children_voxel_start
        left_block_num = ceil(block_nums[axis] / 2)
        left_voxel_stop_axis = left_voxel_start[axis] + \
            left_block_num * leaf_physical_size[axis] // \
                children_voxel_size[axis]
        left_voxel_stop = children_voxel_stop.replace_item(
            axis, left_voxel_stop_axis)
        left_roi = RegionOfInterest(
            left_voxel_start, left_voxel_stop,  children_voxel_size)
        left_tree = cls.from_roi(
            left_roi, factor, leaf_block_size, leaf_voxel_size)

        right_voxel_start = children_voxel_start.replace_item(
            axis, left_voxel_stop_axis
        )
        right_voxel_stop = children_voxel_stop
        right_roi = RegionOfInterest(
            right_voxel_start, 
            right_voxel_stop, 
            children_voxel_size
        )
        right_tree = cls.from_roi(
            right_roi, factor, leaf_block_size, leaf_voxel_size) 

        return cls(roi, axis, left_tree, right_tree)

    @property
    def is_leaf(self):
        return self.axis is None        