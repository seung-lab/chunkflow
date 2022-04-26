from __future__ import annotations

from typing import Union

from .bounding_boxes import Cartesian, BoundingBox


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
        return bbox.to_slices()


class ROIKDTree:
    def __init__(self, roi: RegionOfInterest, dim: int, 
            left: ROIKDTree, right: ROIKDTree):
        """Split the volume hierarchically using K-D tree

        Args:
            roi (RegionOfInterest): physical region covered
            dim (int): the splitting dimension of left and right roi
            left (ROIKDTree): left roi with lower coordinate values in axis dim
            right (ROIKDTree): right roi with higher coordinate values in axis dim
        """
        assert dim >= 0 and dim < 3 
        self.roi = roi
        self.dim = dim
        self.left = left
        self.right = right

    @classmethod
    def from_roi(cls, roi: RegionOfInterest, factor: Cartesian):
        pass
        

    